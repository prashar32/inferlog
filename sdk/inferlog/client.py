"""LoggedLLMClient — the one object the application talks to.

It wraps a set of providers. Every `complete()` / `stream()` call is timed,
its tokens and previews captured, and an InferenceEvent handed to the
dispatcher — on success, on error, and on cancellation alike. The caller
just gets a normal completion or stream back.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator
from uuid import uuid4

from .auto import mark_explicit_logging, unmark_explicit_logging
from .context import current_tags
from .dispatcher import LogDispatcher
from .events import InferenceEvent, utcnow
from .providers import ChatMessage, Completion, Provider, StreamChunk, Usage
from .redaction import Redactor
from .runtime import get_runtime

log = logging.getLogger("inferlog.client")

_PREVIEW_CHARS = 280


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _preview(text: str | None, limit: int) -> str | None:
    text = (text or "").strip()
    return text[:limit] if text else None


def _last_user(messages: list[ChatMessage]) -> str | None:
    return next((m.content for m in reversed(messages) if m.role == "user"), None)


def _classify_error(exc: Exception) -> str:
    """Collapse the zoo of provider exception classes into a stable label.

    The dashboards group on this, so it's worth keeping it small. Worst
    case it lands in 'provider_error', which is still useful.
    """
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return "rate_limit"
    if "timeout" in name:
        return "timeout"
    if "authentication" in name or "permission" in name:
        return "auth"
    if "connection" in name:
        return "connection"
    if "badrequest" in name or "invalidrequest" in name or "notfound" in name:
        return "invalid_request"
    return "provider_error"


class LoggedLLMClient:
    def __init__(
        self,
        *,
        service: str,
        dispatcher: LogDispatcher,
        providers: dict[str, Provider],
        preview_chars: int = _PREVIEW_CHARS,
        redactor: Redactor | None = None,
    ):
        if not providers:
            raise ValueError("LoggedLLMClient needs at least one provider")
        self._service = service
        self._dispatcher = dispatcher
        self._providers = providers
        self._preview_chars = preview_chars
        self._redactor = redactor or Redactor()

    @property
    def providers(self) -> list[str]:
        return sorted(self._providers)

    def _provider(self, name: str) -> Provider:
        try:
            return self._providers[name]
        except KeyError:
            raise ValueError(
                f"unknown provider {name!r}; registered: {self.providers}"
            ) from None

    async def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: list[ChatMessage],
        conversation_id: str | None = None,
        request_id: str | None = None,
        metadata: dict | None = None,
        **opts,
    ) -> Completion:
        prov = self._provider(provider)
        request_id = request_id or str(uuid4())
        started_at = utcnow()
        clock = time.monotonic()
        status, error_type, error_message = "success", None, None
        result: Completion | None = None
        # Tell auto-instrumentation to back off — we're the explicit logger
        # for this call. Prevents double-logging when openai is patched.
        explicit_token = mark_explicit_logging()
        try:
            result = await prov.complete(model, messages, **opts)
            return result
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            status, error_type = "error", _classify_error(exc)
            error_message = str(exc)[:500]
            raise
        finally:
            unmark_explicit_logging(explicit_token)
            self._emit(
                request_id=request_id,
                conversation_id=conversation_id,
                provider=provider,
                model=model,
                status=status,
                streamed=False,
                started_at=started_at,
                latency_ms=_elapsed_ms(clock),
                ttft_ms=None,
                usage=result.usage if result else Usage(),
                input_preview=_preview(_last_user(messages), self._preview_chars),
                output_preview=_preview(result.text if result else None, self._preview_chars),
                error_type=error_type,
                error_message=error_message,
                metadata=metadata or {},
            )

    async def stream(
        self,
        *,
        provider: str,
        model: str,
        messages: list[ChatMessage],
        conversation_id: str | None = None,
        request_id: str | None = None,
        metadata: dict | None = None,
        **opts,
    ) -> AsyncIterator[StreamChunk]:
        prov = self._provider(provider)
        request_id = request_id or str(uuid4())
        started_at = utcnow()
        clock = time.monotonic()
        status, error_type, error_message = "success", None, None
        ttft_ms: int | None = None
        usage = Usage()
        collected: list[str] = []
        explicit_token = mark_explicit_logging()
        try:
            async for chunk in prov.stream(model, messages, **opts):
                if chunk.text:
                    if ttft_ms is None:
                        ttft_ms = _elapsed_ms(clock)
                    collected.append(chunk.text)
                if chunk.usage:
                    usage = chunk.usage
                # Forward any chunk that carries text or usage — the final
                # usage-only chunk matters to the caller too.
                if chunk.text or chunk.usage:
                    yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            # The consumer (gateway) cancelled / closed the stream — e.g. the
            # user hit Stop. We still want a log line, marked accordingly.
            status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            status, error_type = "error", _classify_error(exc)
            error_message = str(exc)[:500]
            raise
        finally:
            self._emit(
                request_id=request_id,
                conversation_id=conversation_id,
                provider=provider,
                model=model,
                status=status,
                streamed=True,
                started_at=started_at,
                latency_ms=_elapsed_ms(clock),
                ttft_ms=ttft_ms,
                usage=usage,
                input_preview=_preview(_last_user(messages), self._preview_chars),
                output_preview=_preview("".join(collected), self._preview_chars),
                error_type=error_type,
                error_message=error_message,
                metadata=metadata or {},
            )
            unmark_explicit_logging(explicit_token)

    def _emit(
        self,
        *,
        request_id: str,
        conversation_id: str | None,
        provider: str,
        model: str,
        status: str,
        streamed: bool,
        started_at,
        latency_ms: int,
        ttft_ms: int | None,
        usage: Usage,
        input_preview: str | None,
        output_preview: str | None,
        error_type: str | None,
        error_message: str | None,
        metadata: dict,
    ) -> None:
        # Redact previews BEFORE the event leaves this process. This is the
        # contract — raw PII never crosses the wire.
        redacted_input, in_n = self._redactor.redact(input_preview)
        redacted_output, out_n = self._redactor.redact(output_preview)
        redacted_error, err_n = self._redactor.redact(error_message)

        tags = current_tags()
        # If a caller used `inferlog.context(conversation_id=...)` and didn't
        # pass conversation_id explicitly, honour the context value.
        if conversation_id is None and "conversation_id" in tags:
            conversation_id = str(tags["conversation_id"])

        event = InferenceEvent(
            request_id=request_id,
            service=self._service,
            provider=provider,
            model=model,
            status=status,
            streamed=streamed,
            started_at=started_at,
            completed_at=utcnow(),
            latency_ms=latency_ms,
            conversation_id=conversation_id,
            ttft_ms=ttft_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            input_preview=redacted_input,
            output_preview=redacted_output,
            error_type=error_type,
            error_message=redacted_error,
            pii_redaction_count=in_n + out_n + err_n,
            tags=tags,
            client_metadata=metadata,
        )
        # Honour the global runtime's sampler if init() was called. The
        # explicit-wrapper path otherwise keeps every event.
        rt = get_runtime()
        if rt is not None and not rt.sampler.should_sample(event):
            self._dispatcher.dropped += 1
            self._dispatcher._notify_drop(1, "sampled")  # type: ignore[attr-defined]
            return
        self._dispatcher.submit(event)
