"""HTTP-level instrumentation — model-agnostic capture by patching httpx.

The contract:
  * Customer calls `inferlog.init(...)` once.
  * Any subsequent LLM call going through httpx — through the official
    `openai` / `anthropic` SDKs, a raw `httpx.AsyncClient`, an OpenAI-
    compatible proxy (vLLM / OpenRouter / Together / LiteLLM), an Ollama
    server, or any other HTTP-speaking model — is captured.
  * The customer's call sites are untouched. They never name the
    provider; we figure it out from the URL.

Why HTTP-level rather than per-library: library patches couple us to
each vendor's Python SDK internals (and break when those SDKs refactor).
HTTP capture sits below all of them — the only contract we depend on is
"the request leaves the process via httpx".

Two safety rules, enforced everywhere:
  1. If our capture logic throws, the host's call MUST still work. Every
     code path falls back to calling the original `send`.
  2. The `LoggedLLMClient` (explicit-wrapper) path sets a contextvar so
     our HTTP wrapper backs off — preventing double-logging when the
     wrapper internally talks to httpx.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from typing import Any
from uuid import uuid4

import httpx

from .context import current_tags
from .events import InferenceEvent, utcnow
from .parsers import HANDLERS, ProviderHandler, RequestMeta, ResponseMeta
from .runtime import Runtime, get_runtime as _get_global_runtime

log = logging.getLogger("inferlog.auto")
_PREVIEW_CHARS = 280

# Set inside LoggedLLMClient so HTTP capture stands down inside the
# explicit-wrapper scope.
_inside_explicit: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "inferlog_inside_explicit", default=False
)


def mark_explicit_logging() -> contextvars.Token:
    return _inside_explicit.set(True)


def unmark_explicit_logging(token: contextvars.Token) -> None:
    _inside_explicit.reset(token)


# ---------------------------------------------------------------- install


# Module-level state. Read at call time so a second `init()` cleanly
# swaps the active runtime without rebinding the class methods.
_state: dict[str, Any] = {}


def install(runtime: Runtime) -> list[str]:
    """Patch `httpx.AsyncClient.send` and `httpx.Client.send`. Idempotent."""
    if _state.get("installed"):
        _state["runtime"] = runtime
        return ["httpx"]
    _state["runtime"] = runtime
    _state["async_send"] = httpx.AsyncClient.send
    _state["sync_send"] = httpx.Client.send

    async def patched_async_send(self, request, **kwargs):
        return await _capture_async(self, request, kwargs)

    def patched_sync_send(self, request, **kwargs):
        return _capture_sync(self, request, kwargs)

    httpx.AsyncClient.send = patched_async_send  # type: ignore[assignment]
    httpx.Client.send = patched_sync_send  # type: ignore[assignment]
    _state["installed"] = True
    log.info("inferlog: patched httpx.AsyncClient.send and httpx.Client.send")
    return ["httpx"]


def uninstall() -> None:
    """Restore the originals. Mostly for tests."""
    if not _state.get("installed"):
        return
    httpx.AsyncClient.send = _state["async_send"]  # type: ignore[assignment]
    httpx.Client.send = _state["sync_send"]  # type: ignore[assignment]
    _state.clear()


def is_global_install_active() -> bool:
    """True when `httpx.send` is globally patched. The transport uses
    this to step aside if global capture is already running (avoids
    double-logging if a customer pairs both modes)."""
    return bool(_state.get("installed"))


# ---------------------------------------------------------------- helpers


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _classify_error(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return "rate_limit"
    if "timeout" in name:
        return "timeout"
    if "auth" in name or "permission" in name:
        return "auth"
    if "connect" in name:
        return "connection"
    return "provider_error"


def _classify_status(code: int) -> str:
    if code in (401, 403):
        return "auth"
    if code == 429:
        return "rate_limit"
    if code in (408, 504):
        return "timeout"
    if 400 <= code < 500:
        return "invalid_request"
    return "provider_error"


def _identify(request: httpx.Request) -> ProviderHandler | None:
    for h in HANDLERS:
        try:
            if h.matches(request):
                return h
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------- async path


async def _capture_async(self, request, kwargs):
    rt: Runtime | None = _get_global_runtime()
    original = _state["async_send"]

    # Three escape hatches — fast path through.
    if rt is None or not rt.enabled or _inside_explicit.get():
        return await original(self, request, **kwargs)

    handler = _identify(request)
    if handler is None:
        # Not an LLM call we recognise — never touch it.
        return await original(self, request, **kwargs)

    try:
        request_meta = handler.parse_request(request)
    except Exception:  # noqa: BLE001
        log.exception("inferlog: parse_request failed for %s; passing through", handler.name)
        return await original(self, request, **kwargs)

    request_id = str(uuid4())
    started_at = utcnow()
    clock = time.monotonic()

    # Network / protocol error → emit error event, re-raise.
    try:
        response = await original(self, request, **kwargs)
    except Exception as exc:
        _emit_safe(
            rt, request_id=request_id, handler=handler, request_meta=request_meta,
            status="error", started_at=started_at, latency_ms=_elapsed_ms(clock),
            error_type=_classify_error(exc), error_message=str(exc)[:500],
        )
        raise

    # HTTP-level error response — read body, classify, emit.
    if response.status_code >= 400:
        err_msg = f"HTTP {response.status_code}"
        try:
            if not response.is_closed:
                await response.aread()
            err_msg = (response.text or "")[:500] or err_msg
        except Exception:  # noqa: BLE001
            pass
        _emit_safe(
            rt, request_id=request_id, handler=handler, request_meta=request_meta,
            status="error", started_at=started_at, latency_ms=_elapsed_ms(clock),
            error_type=_classify_status(response.status_code), error_message=err_msg,
        )
        return response

    # Non-streaming success — body is buffered, parse it now.
    if not request_meta.streaming:
        try:
            if not response.is_closed:
                await response.aread()
            response_meta = handler.parse_response(response)
        except Exception:  # noqa: BLE001
            log.exception("inferlog: parse_response failed for %s", handler.name)
            response_meta = ResponseMeta()
        _emit_safe(
            rt, request_id=request_id, handler=handler, request_meta=request_meta,
            status="success", started_at=started_at, latency_ms=_elapsed_ms(clock),
            response_meta=response_meta,
        )
        return response

    # Streaming — wrap the response iterators. Emission happens when the
    # customer's `async for` finishes (or is cancelled / errors).
    _attach_async_stream_capture(
        response, rt, request_id, handler, request_meta, started_at, clock,
    )
    return response


def _attach_async_stream_capture(
    response, rt, request_id, handler, request_meta, started_at, clock,
):
    """Wrap a streaming response so we observe each chunk as the consumer
    iterates it.

    We only wrap `aiter_bytes`. httpx's other iterator methods funnel
    down to it (`aiter_lines` → `aiter_bytes`, `aiter_text` → `aiter_bytes`),
    so a single wrap captures every consumer path exactly once. Wrapping
    multiple methods caused double-feeding because `aiter_bytes` itself
    calls `aiter_raw` internally — if both are wrapped, the same bytes
    flow through the parser twice.
    """
    parser = handler.make_stream_parser()
    state = _AsyncStreamState(
        rt, request_id, handler, request_meta, started_at, clock, parser,
    )

    orig_bytes = response.aiter_bytes

    async def wrapped_aiter_bytes(*args, **kwargs):
        try:
            async for chunk in orig_bytes(*args, **kwargs):
                state.feed(chunk)
                yield chunk
            state.emit("success")
        except (asyncio.CancelledError, GeneratorExit):
            # If the parser saw the natural completion marker before the
            # consumer broke out of iteration, treat that as success.
            # Library wrappers (e.g. the openai SDK) routinely break
            # early once they parse `[DONE]`.
            state.emit("success" if state.completed else "cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            state.emit("error", exc=exc)
            raise

    response.aiter_bytes = wrapped_aiter_bytes  # type: ignore[assignment]


class _AsyncStreamState:
    """Holds the per-stream observation state. Exactly one `emit()` call
    per stream — guarded by `_emitted`."""

    @property
    def completed(self) -> bool:
        return self._parser.completed


    def __init__(self, rt, request_id, handler, request_meta, started_at, clock, parser):
        self._rt = rt
        self._request_id = request_id
        self._handler = handler
        self._request_meta = request_meta
        self._started_at = started_at
        self._clock = clock
        self._parser = parser
        self._ttft_ms: int | None = None
        self._emitted = False

    def feed(self, chunk: bytes) -> None:
        try:
            if not chunk:
                return
            had_output_before = bool(self._parser.output_text)
            self._parser.feed(chunk)
            if (
                not had_output_before
                and self._parser.output_text
                and self._ttft_ms is None
            ):
                self._ttft_ms = _elapsed_ms(self._clock)
        except Exception:  # noqa: BLE001
            log.exception("inferlog stream parse failed (swallowed)")

    def emit(self, status: str, exc: BaseException | None = None) -> None:
        if self._emitted:
            return
        self._emitted = True
        _emit_safe(
            self._rt,
            request_id=self._request_id,
            handler=self._handler,
            request_meta=self._request_meta,
            status=status,
            started_at=self._started_at,
            latency_ms=_elapsed_ms(self._clock),
            ttft_ms=self._ttft_ms,
            response_meta=ResponseMeta(
                output_text=self._parser.output_text or None,
                prompt_tokens=self._parser.prompt_tokens,
                completion_tokens=self._parser.completion_tokens,
                total_tokens=self._parser.total_tokens,
            ),
            error_type=_classify_error(exc) if exc else None,
            error_message=str(exc)[:500] if exc else None,
        )


# ---------------------------------------------------------------- sync path


def _capture_sync(self, request, kwargs):
    rt: Runtime | None = _get_global_runtime()
    original = _state["sync_send"]

    if rt is None or not rt.enabled or _inside_explicit.get():
        return original(self, request, **kwargs)

    handler = _identify(request)
    if handler is None:
        return original(self, request, **kwargs)

    try:
        request_meta = handler.parse_request(request)
    except Exception:  # noqa: BLE001
        return original(self, request, **kwargs)

    request_id = str(uuid4())
    started_at = utcnow()
    clock = time.monotonic()

    try:
        response = original(self, request, **kwargs)
    except Exception as exc:
        _emit_safe(
            rt, request_id=request_id, handler=handler, request_meta=request_meta,
            status="error", started_at=started_at, latency_ms=_elapsed_ms(clock),
            error_type=_classify_error(exc), error_message=str(exc)[:500],
        )
        raise

    if response.status_code >= 400:
        err_msg = f"HTTP {response.status_code}"
        try:
            if not response.is_closed:
                response.read()
            err_msg = (response.text or "")[:500] or err_msg
        except Exception:  # noqa: BLE001
            pass
        _emit_safe(
            rt, request_id=request_id, handler=handler, request_meta=request_meta,
            status="error", started_at=started_at, latency_ms=_elapsed_ms(clock),
            error_type=_classify_status(response.status_code), error_message=err_msg,
        )
        return response

    if not request_meta.streaming:
        try:
            if not response.is_closed:
                response.read()
            response_meta = handler.parse_response(response)
        except Exception:  # noqa: BLE001
            log.exception("inferlog: parse_response failed (sync)")
            response_meta = ResponseMeta()
        _emit_safe(
            rt, request_id=request_id, handler=handler, request_meta=request_meta,
            status="success", started_at=started_at, latency_ms=_elapsed_ms(clock),
            response_meta=response_meta,
        )
        return response

    # Sync streaming: emit a header-only event now (no token counts /
    # output preview). The next iteration is a sync-stream wrapper; sync
    # streaming is uncommon in LLM apps so this is acceptable for v0.2.
    log.debug("inferlog: sync streaming response not wrapped (headers-only event)")
    _emit_safe(
        rt, request_id=request_id, handler=handler, request_meta=request_meta,
        status="success", started_at=started_at, latency_ms=_elapsed_ms(clock),
        response_meta=ResponseMeta(),
    )
    return response


# ---------------------------------------------------------------- emit


def _emit_safe(
    rt: Runtime,
    *,
    request_id: str,
    handler: ProviderHandler,
    request_meta: RequestMeta,
    status: str,
    started_at,
    latency_ms: int,
    ttft_ms: int | None = None,
    response_meta: ResponseMeta | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Apply redaction + sampling and submit. Wrapped in try/except so a
    bug here never breaks the customer's call."""
    try:
        response_meta = response_meta or ResponseMeta()
        redactor = rt.redactor

        input_preview = (request_meta.input_text or "").strip()[:_PREVIEW_CHARS] or None
        output_preview = (response_meta.output_text or "").strip()[:_PREVIEW_CHARS] or None

        red_input, in_n = redactor.redact(input_preview)
        red_output, out_n = redactor.redact(output_preview)
        red_error, err_n = redactor.redact(error_message)

        tags = current_tags()
        conversation_id = tags.get("conversation_id")
        if conversation_id is not None:
            conversation_id = str(conversation_id)

        event = InferenceEvent(
            request_id=request_id,
            service=rt.service,
            provider=handler.name,
            model=request_meta.model,
            status=status,
            streamed=request_meta.streaming,
            started_at=started_at,
            completed_at=utcnow(),
            latency_ms=latency_ms,
            conversation_id=conversation_id,
            ttft_ms=ttft_ms,
            prompt_tokens=response_meta.prompt_tokens,
            completion_tokens=response_meta.completion_tokens,
            total_tokens=response_meta.total_tokens,
            input_preview=red_input,
            output_preview=red_output,
            error_type=error_type,
            error_message=red_error,
            pii_redaction_count=in_n + out_n + err_n,
            tags=tags,
            client_metadata={"auto_instrumented": True, "transport": "httpx"},
        )
        if not rt.sampler.should_sample(event):
            rt.dispatcher.dropped += 1
            rt.dispatcher._notify_drop(1, "sampled")  # type: ignore[attr-defined]
            return
        rt.dispatcher.submit(event)
    except Exception:  # noqa: BLE001
        log.exception("inferlog auto-emit failed (swallowed)")


# ---------------------------------------------------------------- transport
# An httpx Transport that captures LLM calls WITHOUT globally patching
# httpx. The customer attaches it per-client; only requests through that
# client touch our code. Other httpx clients in the same process are
# completely untouched.
#
#     client = httpx.AsyncClient(transport=inferlog.transport())
#     oai = openai.AsyncOpenAI(http_client=httpx.AsyncClient(transport=inferlog.transport()))


class _CapturingAsyncByteStream(httpx.AsyncByteStream):
    """Wraps an httpx response body stream so that — as the consumer
    iterates it — our parser observes each chunk and emits when the
    iteration finishes (or is cancelled / errors)."""

    def __init__(self, inner: httpx.AsyncByteStream, state: "_AsyncStreamState"):
        self._inner = inner
        self._state = state

    async def __aiter__(self):
        try:
            async for chunk in self._inner:
                self._state.feed(chunk)
                yield chunk
            self._state.emit("success")
        except (asyncio.CancelledError, GeneratorExit):
            self._state.emit(
                "success" if self._state.completed else "cancelled"
            )
            raise
        except Exception as exc:  # noqa: BLE001
            self._state.emit("error", exc=exc)
            raise

    async def aclose(self) -> None:
        await self._inner.aclose()


class InferLogAsyncTransport(httpx.AsyncBaseTransport):
    """An httpx transport that instruments LLM calls only.

    Attach to a specific client to opt that client into inferlog capture
    without globally patching httpx:

        client = httpx.AsyncClient(transport=inferlog.transport())

    No other httpx clients in the process are affected. If global httpx
    patching is also active, this transport steps aside to avoid double-
    logging — pick one mode or the other per client.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None):
        self._inner = inner if inner is not None else httpx.AsyncHTTPTransport()

    async def __aenter__(self) -> "InferLogAsyncTransport":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._inner.__aexit__(*exc_info)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        rt: Runtime | None = _get_global_runtime()

        # Step aside when there's no runtime, when capture is disabled,
        # when the global httpx patch is already active (it'll capture
        # at the higher level), or when an explicit-wrapper scope is on.
        if (
            rt is None
            or not rt.enabled
            or _state.get("installed")
            or _inside_explicit.get()
        ):
            return await self._inner.handle_async_request(request)

        handler = _identify(request)
        if handler is None:
            return await self._inner.handle_async_request(request)

        try:
            request_meta = handler.parse_request(request)
        except Exception:  # noqa: BLE001
            log.exception("inferlog: parse_request failed for %s; passing through", handler.name)
            return await self._inner.handle_async_request(request)

        request_id = str(uuid4())
        started_at = utcnow()
        clock = time.monotonic()

        try:
            response = await self._inner.handle_async_request(request)
        except Exception as exc:
            _emit_safe(
                rt, request_id=request_id, handler=handler, request_meta=request_meta,
                status="error", started_at=started_at, latency_ms=_elapsed_ms(clock),
                error_type=_classify_error(exc), error_message=str(exc)[:500],
            )
            raise

        if response.status_code >= 400:
            err_msg = f"HTTP {response.status_code}"
            try:
                await response.aread()
                err_msg = (response.text or "")[:500] or err_msg
            except Exception:  # noqa: BLE001
                pass
            _emit_safe(
                rt, request_id=request_id, handler=handler, request_meta=request_meta,
                status="error", started_at=started_at, latency_ms=_elapsed_ms(clock),
                error_type=_classify_status(response.status_code), error_message=err_msg,
            )
            return response

        if not request_meta.streaming:
            try:
                await response.aread()
                response_meta = handler.parse_response(response)
            except Exception:  # noqa: BLE001
                log.exception("inferlog: parse_response failed for %s", handler.name)
                response_meta = ResponseMeta()
            _emit_safe(
                rt, request_id=request_id, handler=handler, request_meta=request_meta,
                status="success", started_at=started_at, latency_ms=_elapsed_ms(clock),
                response_meta=response_meta,
            )
            return response

        # Streaming: wrap the response's underlying stream. Whichever
        # iterator method the consumer reaches for (`aiter_bytes`,
        # `aiter_lines`, etc.) ultimately drains this stream — and our
        # wrapper observes every chunk on the way through.
        parser = handler.make_stream_parser()
        state = _AsyncStreamState(
            rt, request_id, handler, request_meta, started_at, clock, parser,
        )
        wrapped = _CapturingAsyncByteStream(response.stream, state)  # type: ignore[arg-type]
        # Note: don't carry `response.request` — httpx assigns that after
        # the transport returns. Pass the original `request` we received.
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=wrapped,
            request=request,
            extensions=response.extensions,
        )


def transport(
    inner: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncBaseTransport:
    """Return an httpx transport that captures LLM calls on whatever
    client it's attached to. Use this when you want surgical capture
    instead of `init(capture_all_httpx=True)`'s process-wide patching."""
    return InferLogAsyncTransport(inner)
