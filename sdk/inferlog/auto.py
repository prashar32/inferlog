"""Auto-instrumentation — monkey-patch popular LLM client libraries so
inference logging is automatic.

The contract: a customer calls `inferlog.init(api_key=..., endpoint=...)`
once and the rest of their code is unchanged. Any subsequent call through
`openai`, `anthropic`, etc. is captured and shipped — same as New Relic
or Sentry's pattern.

Two safety rules in every wrapper:
  1. If our capture logic throws, we MUST NOT break the host call. Log and
     pass through. Observability that breaks the app is worse than no
     observability.
  2. If the caller is inside the legacy LoggedLLMClient path, skip
     capturing here so we don't double-log.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from typing import Any
from uuid import uuid4

from .context import current_tags
from .events import InferenceEvent, utcnow
from .runtime import Runtime, get_runtime

log = logging.getLogger("inferlog.auto")

_PREVIEW_CHARS = 280

# Set inside LoggedLLMClient so auto-instrumentation knows to back off.
_inside_explicit: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "inferlog_inside_explicit", default=False
)


def mark_explicit_logging() -> contextvars.Token:
    """Used by LoggedLLMClient to suppress auto-capture inside its scope."""
    return _inside_explicit.set(True)


def unmark_explicit_logging(token: contextvars.Token) -> None:
    _inside_explicit.reset(token)


def install(runtime: Runtime) -> list[str]:
    """Install every auto-instrumentation we know about.

    Returns the providers actually instrumented. Idempotent — calling
    twice is safe.
    """
    installed: list[str] = []
    if _install_openai(runtime):
        installed.append("openai")
    if _install_anthropic(runtime):
        installed.append("anthropic")
    return installed


def uninstall() -> None:
    """Undo every patch. Mostly for tests."""
    _uninstall_openai()
    _uninstall_anthropic()


# ----------------------------------------------------------------------
# Shared emission helper
# ----------------------------------------------------------------------


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _truncate(text: str | None, limit: int = _PREVIEW_CHARS) -> str | None:
    if text is None:
        return None
    text = text.strip()
    return text[:limit] if text else None


def _classify_error(exc: BaseException) -> str:
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


def _emit(
    runtime: Runtime,
    *,
    request_id: str,
    provider: str,
    model: str,
    status: str,
    streamed: bool,
    started_at,
    latency_ms: int,
    ttft_ms: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Apply redaction, sampling, and submit. Wrapped in try/except — if
    anything here raises we swallow and log; we will not break the
    customer's call."""
    try:
        redactor = runtime.redactor
        redacted_input, in_n = redactor.redact(_truncate(input_text))
        redacted_output, out_n = redactor.redact(_truncate(output_text))
        redacted_error, err_n = redactor.redact(error_message)

        tags = current_tags()
        conversation_id = tags.get("conversation_id")
        if conversation_id is not None:
            conversation_id = str(conversation_id)

        event = InferenceEvent(
            request_id=request_id,
            service=runtime.service,
            provider=provider,
            model=model,
            status=status,
            streamed=streamed,
            started_at=started_at,
            completed_at=utcnow(),
            latency_ms=latency_ms,
            conversation_id=conversation_id,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_preview=redacted_input,
            output_preview=redacted_output,
            error_type=error_type,
            error_message=redacted_error,
            pii_redaction_count=in_n + out_n + err_n,
            tags=tags,
            client_metadata={"auto_instrumented": True},
        )
        if not runtime.sampler.should_sample(event):
            runtime.dispatcher.dropped += 1
            runtime.dispatcher._notify_drop(1, "sampled")  # type: ignore[attr-defined]
            return
        runtime.dispatcher.submit(event)
    except Exception:  # noqa: BLE001
        log.exception("inferlog auto-emit failed (swallowed)")


def _last_user_text(messages: Any) -> str | None:
    """Pull the last user-role content out of an OpenAI/Anthropic-shaped
    message list. Defensive — never raise on unexpected shapes."""
    try:
        for msg in reversed(messages or []):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "user":
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # multimodal content list of parts
                    return " ".join(
                        p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                        for p in content
                    ).strip() or None
    except Exception:  # noqa: BLE001
        return None
    return None


# ----------------------------------------------------------------------
# OpenAI
# ----------------------------------------------------------------------


_openai_original: dict[str, Any] = {}


def _install_openai(runtime: Runtime) -> bool:
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except ImportError:
        return False

    if _openai_original.get("installed"):
        # Re-running install: keep existing patches but rebind runtime.
        _openai_original["runtime"] = runtime
        return True

    _openai_original["runtime"] = runtime
    _openai_original["async_create"] = AsyncCompletions.create
    _openai_original["sync_create"] = Completions.create

    async def patched_async_create(self, *args, **kwargs):
        rt = _openai_original["runtime"]
        if rt is None or not rt.enabled or _inside_explicit.get():
            return await _openai_original["async_create"](self, *args, **kwargs)
        return await _capture_openai_async_create(rt, self, args, kwargs)

    def patched_sync_create(self, *args, **kwargs):
        rt = _openai_original["runtime"]
        if rt is None or not rt.enabled or _inside_explicit.get():
            return _openai_original["sync_create"](self, *args, **kwargs)
        return _capture_openai_sync_create(rt, self, args, kwargs)

    AsyncCompletions.create = patched_async_create  # type: ignore[assignment]
    Completions.create = patched_sync_create  # type: ignore[assignment]
    _openai_original["installed"] = True
    log.info("inferlog: patched openai chat.completions.create (sync + async)")
    return True


def _uninstall_openai() -> None:
    if not _openai_original.get("installed"):
        return
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
        AsyncCompletions.create = _openai_original["async_create"]  # type: ignore[assignment]
        Completions.create = _openai_original["sync_create"]  # type: ignore[assignment]
    except ImportError:
        pass
    _openai_original.clear()


async def _capture_openai_async_create(runtime: Runtime, self, args, kwargs):
    model = kwargs.get("model", "unknown")
    streaming = bool(kwargs.get("stream"))
    # Force usage emission on streams so we can capture token counts.
    if streaming and "stream_options" not in kwargs:
        kwargs["stream_options"] = {"include_usage": True}

    started_at = utcnow()
    clock = time.monotonic()
    request_id = str(uuid4())
    input_text = _last_user_text(kwargs.get("messages"))
    original = _openai_original["async_create"]

    try:
        result = await original(self, *args, **kwargs)
    except Exception as exc:
        _emit(
            runtime,
            request_id=request_id, provider="openai", model=model,
            status="error", streamed=streaming,
            started_at=started_at, latency_ms=_elapsed_ms(clock),
            error_type=_classify_error(exc), error_message=str(exc)[:500],
            input_text=input_text,
        )
        raise

    if not streaming:
        usage = getattr(result, "usage", None)
        _emit(
            runtime,
            request_id=request_id, provider="openai", model=model,
            status="success", streamed=False,
            started_at=started_at, latency_ms=_elapsed_ms(clock),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            input_text=input_text,
            output_text=_openai_extract_message_text(result),
        )
        return result

    return _OpenAIAsyncStreamWrapper(
        result, runtime=runtime, request_id=request_id, model=model,
        started_at=started_at, clock=clock, input_text=input_text,
    )


def _capture_openai_sync_create(runtime: Runtime, self, args, kwargs):
    # Mirror of the async path. Streams aren't wrapped — sync streaming is
    # uncommon and the wrapper is more involved. If you need it, ask.
    model = kwargs.get("model", "unknown")
    streaming = bool(kwargs.get("stream"))
    started_at = utcnow()
    clock = time.monotonic()
    request_id = str(uuid4())
    input_text = _last_user_text(kwargs.get("messages"))
    original = _openai_original["sync_create"]
    try:
        result = original(self, *args, **kwargs)
    except Exception as exc:
        _emit(
            runtime,
            request_id=request_id, provider="openai", model=model,
            status="error", streamed=streaming,
            started_at=started_at, latency_ms=_elapsed_ms(clock),
            error_type=_classify_error(exc), error_message=str(exc)[:500],
            input_text=input_text,
        )
        raise
    if not streaming:
        usage = getattr(result, "usage", None)
        _emit(
            runtime,
            request_id=request_id, provider="openai", model=model,
            status="success", streamed=False,
            started_at=started_at, latency_ms=_elapsed_ms(clock),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            input_text=input_text,
            output_text=_openai_extract_message_text(result),
        )
    return result


def _openai_extract_message_text(response: Any) -> str | None:
    try:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        msg = getattr(choices[0], "message", None)
        return getattr(msg, "content", None) if msg is not None else None
    except Exception:  # noqa: BLE001
        return None


class _OpenAIAsyncStreamWrapper:
    """Async-iterable proxy around an openai AsyncStream.

    Supports `async for`, `async with`, and `.close()` — the three patterns
    customers actually use with openai's stream object.
    """

    def __init__(
        self, inner, *, runtime: Runtime, request_id: str, model: str,
        started_at, clock: float, input_text: str | None,
    ):
        self._inner = inner
        self._runtime = runtime
        self._request_id = request_id
        self._model = model
        self._started_at = started_at
        self._clock = clock
        self._input_text = input_text
        self._collected: list[str] = []
        self._usage = None
        self._ttft_ms: int | None = None
        self._emitted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._inner.__anext__()
        except StopAsyncIteration:
            self._emit("success")
            raise
        except (asyncio.CancelledError, GeneratorExit):
            self._emit("cancelled")
            raise
        except Exception as exc:
            self._emit("error", exc=exc)
            raise
        self._observe(chunk)
        return chunk

    def _observe(self, chunk) -> None:
        try:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    if self._ttft_ms is None:
                        self._ttft_ms = _elapsed_ms(self._clock)
                    self._collected.append(content)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._usage = usage
        except Exception:  # noqa: BLE001
            pass

    def _emit(self, status: str, exc: BaseException | None = None) -> None:
        if self._emitted:
            return
        self._emitted = True
        _emit(
            self._runtime,
            request_id=self._request_id,
            provider="openai",
            model=self._model,
            status=status,
            streamed=True,
            started_at=self._started_at,
            latency_ms=_elapsed_ms(self._clock),
            ttft_ms=self._ttft_ms,
            prompt_tokens=getattr(self._usage, "prompt_tokens", None),
            completion_tokens=getattr(self._usage, "completion_tokens", None),
            total_tokens=getattr(self._usage, "total_tokens", None),
            input_text=self._input_text,
            output_text="".join(self._collected) or None,
            error_type=_classify_error(exc) if exc else None,
            error_message=str(exc)[:500] if exc else None,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._emitted:
            self._emit("error" if exc_type else "success")
        if hasattr(self._inner, "close"):
            try:
                await self._inner.close()
            except Exception:  # noqa: BLE001
                pass
        return False

    async def close(self) -> None:
        if hasattr(self._inner, "close"):
            try:
                await self._inner.close()
            except Exception:  # noqa: BLE001
                pass
        if not self._emitted:
            self._emit("cancelled")


# ----------------------------------------------------------------------
# Anthropic — similar shape; their stream events differ.
# ----------------------------------------------------------------------


_anthropic_original: dict[str, Any] = {}


def _install_anthropic(runtime: Runtime) -> bool:
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except ImportError:
        return False
    if _anthropic_original.get("installed"):
        _anthropic_original["runtime"] = runtime
        return True

    _anthropic_original["runtime"] = runtime
    _anthropic_original["async_create"] = AsyncMessages.create
    _anthropic_original["sync_create"] = Messages.create

    async def patched_async_create(self, *args, **kwargs):
        rt = _anthropic_original["runtime"]
        if rt is None or not rt.enabled or _inside_explicit.get():
            return await _anthropic_original["async_create"](self, *args, **kwargs)
        return await _capture_anthropic_async_create(rt, self, args, kwargs)

    def patched_sync_create(self, *args, **kwargs):
        rt = _anthropic_original["runtime"]
        if rt is None or not rt.enabled or _inside_explicit.get():
            return _anthropic_original["sync_create"](self, *args, **kwargs)
        # sync streaming is left to the original — capture only non-stream
        return _capture_anthropic_sync_create(rt, self, args, kwargs)

    AsyncMessages.create = patched_async_create  # type: ignore[assignment]
    Messages.create = patched_sync_create  # type: ignore[assignment]
    _anthropic_original["installed"] = True
    log.info("inferlog: patched anthropic messages.create (sync + async)")
    return True


def _uninstall_anthropic() -> None:
    if not _anthropic_original.get("installed"):
        return
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
        AsyncMessages.create = _anthropic_original["async_create"]  # type: ignore[assignment]
        Messages.create = _anthropic_original["sync_create"]  # type: ignore[assignment]
    except ImportError:
        pass
    _anthropic_original.clear()


async def _capture_anthropic_async_create(runtime: Runtime, self, args, kwargs):
    model = kwargs.get("model", "unknown")
    streaming = bool(kwargs.get("stream"))
    started_at = utcnow()
    clock = time.monotonic()
    request_id = str(uuid4())
    input_text = _last_user_text(kwargs.get("messages"))
    original = _anthropic_original["async_create"]

    try:
        result = await original(self, *args, **kwargs)
    except Exception as exc:
        _emit(
            runtime,
            request_id=request_id, provider="anthropic", model=model,
            status="error", streamed=streaming,
            started_at=started_at, latency_ms=_elapsed_ms(clock),
            error_type=_classify_error(exc), error_message=str(exc)[:500],
            input_text=input_text,
        )
        raise

    if not streaming:
        usage = getattr(result, "usage", None)
        text = _anthropic_extract_text(result)
        prompt = getattr(usage, "input_tokens", None)
        completion = getattr(usage, "output_tokens", None)
        _emit(
            runtime,
            request_id=request_id, provider="anthropic", model=model,
            status="success", streamed=False,
            started_at=started_at, latency_ms=_elapsed_ms(clock),
            prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=(prompt + completion) if (prompt and completion) else None,
            input_text=input_text, output_text=text,
        )
        return result

    return _AnthropicAsyncStreamWrapper(
        result, runtime=runtime, request_id=request_id, model=model,
        started_at=started_at, clock=clock, input_text=input_text,
    )


def _capture_anthropic_sync_create(runtime: Runtime, self, args, kwargs):
    model = kwargs.get("model", "unknown")
    streaming = bool(kwargs.get("stream"))
    started_at = utcnow()
    clock = time.monotonic()
    request_id = str(uuid4())
    input_text = _last_user_text(kwargs.get("messages"))
    original = _anthropic_original["sync_create"]
    try:
        result = original(self, *args, **kwargs)
    except Exception as exc:
        _emit(
            runtime,
            request_id=request_id, provider="anthropic", model=model,
            status="error", streamed=streaming,
            started_at=started_at, latency_ms=_elapsed_ms(clock),
            error_type=_classify_error(exc), error_message=str(exc)[:500],
            input_text=input_text,
        )
        raise
    if not streaming:
        usage = getattr(result, "usage", None)
        prompt = getattr(usage, "input_tokens", None)
        completion = getattr(usage, "output_tokens", None)
        _emit(
            runtime,
            request_id=request_id, provider="anthropic", model=model,
            status="success", streamed=False,
            started_at=started_at, latency_ms=_elapsed_ms(clock),
            prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=(prompt + completion) if (prompt and completion) else None,
            input_text=input_text, output_text=_anthropic_extract_text(result),
        )
    return result


def _anthropic_extract_text(response: Any) -> str | None:
    try:
        blocks = getattr(response, "content", None) or []
        parts = []
        for b in blocks:
            if getattr(b, "type", None) == "text":
                parts.append(getattr(b, "text", "") or "")
        return "".join(parts) or None
    except Exception:  # noqa: BLE001
        return None


class _AnthropicAsyncStreamWrapper:
    """Wraps Anthropic's AsyncMessageStream. Their streaming protocol is
    event-based (`message_start`, `content_block_delta`, `message_delta`),
    so the capture logic differs from OpenAI."""

    def __init__(
        self, inner, *, runtime: Runtime, request_id: str, model: str,
        started_at, clock: float, input_text: str | None,
    ):
        self._inner = inner
        self._runtime = runtime
        self._request_id = request_id
        self._model = model
        self._started_at = started_at
        self._clock = clock
        self._input_text = input_text
        self._collected: list[str] = []
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None
        self._ttft_ms: int | None = None
        self._emitted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._inner.__anext__()
        except StopAsyncIteration:
            self._emit("success")
            raise
        except (asyncio.CancelledError, GeneratorExit):
            self._emit("cancelled")
            raise
        except Exception as exc:
            self._emit("error", exc=exc)
            raise
        self._observe(event)
        return event

    def _observe(self, event) -> None:
        try:
            etype = getattr(event, "type", None)
            if etype == "message_start":
                msg = getattr(event, "message", None)
                usage = getattr(msg, "usage", None) if msg else None
                if usage is not None:
                    self._prompt_tokens = getattr(usage, "input_tokens", None)
            elif etype == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "type", None) == "text_delta":
                    text = getattr(delta, "text", "") or ""
                    if text:
                        if self._ttft_ms is None:
                            self._ttft_ms = _elapsed_ms(self._clock)
                        self._collected.append(text)
            elif etype == "message_delta":
                usage = getattr(event, "usage", None)
                if usage is not None:
                    self._completion_tokens = getattr(usage, "output_tokens", None)
        except Exception:  # noqa: BLE001
            pass

    def _emit(self, status: str, exc: BaseException | None = None) -> None:
        if self._emitted:
            return
        self._emitted = True
        total = None
        if self._prompt_tokens is not None and self._completion_tokens is not None:
            total = self._prompt_tokens + self._completion_tokens
        _emit(
            self._runtime,
            request_id=self._request_id,
            provider="anthropic",
            model=self._model,
            status=status,
            streamed=True,
            started_at=self._started_at,
            latency_ms=_elapsed_ms(self._clock),
            ttft_ms=self._ttft_ms,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=total,
            input_text=self._input_text,
            output_text="".join(self._collected) or None,
            error_type=_classify_error(exc) if exc else None,
            error_message=str(exc)[:500] if exc else None,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._emitted:
            self._emit("error" if exc_type else "success")
        if hasattr(self._inner, "close"):
            try:
                await self._inner.close()
            except Exception:  # noqa: BLE001
                pass
        return False

    async def close(self) -> None:
        if hasattr(self._inner, "close"):
            try:
                await self._inner.close()
            except Exception:  # noqa: BLE001
                pass
        if not self._emitted:
            self._emit("cancelled")
