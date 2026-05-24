"""inferlog — turn any LLM call into a structured inference log without
the caller having to think about it.

Two ways to integrate:

  1. **One-line auto-instrumentation** (recommended for most apps):

         import inferlog
         inferlog.init(api_key=..., endpoint="https://ingest.example.com/v1/ingest")

         # Existing OpenAI / Anthropic code is unchanged and now logged:
         resp = await openai_client.chat.completions.create(...)

         # Optional: tag a scope (no per-call code change)
         with inferlog.context(conversation_id=cid, user_id=uid):
             ...

  2. **Explicit wrapper** (for custom providers or when you want manual
     control):

         from inferlog import LoggedLLMClient, LogDispatcher, HttpSink
         dispatcher = LogDispatcher(HttpSink("http://..."))
         dispatcher.start()
         client = LoggedLLMClient(service=..., dispatcher=dispatcher,
                                  providers={"mock": MockProvider()})
         async for chunk in client.stream(...): ...

PII redaction runs INSIDE the host process for both paths — raw bytes
never cross the wire.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
from typing import Any, Callable
from urllib.parse import urlparse

from . import auto as _auto
from .client import LoggedLLMClient
from .context import context, current_tags
from .dispatcher import (
    HttpSink,
    LogDispatcher,
    MemorySink,
    NullSink,
    TransientDeliveryError,
)
from .events import SDK_VERSION, InferenceEvent
from .providers import ChatMessage, Completion, StreamChunk, Usage
from .redaction import Redactor
from .runtime import (
    Runtime,
    build_default_runtime,
    get_runtime,
    is_initialized,
    set_runtime,
)
from .sampling import AlwaysKeepErrors, CustomSampler, KeepAll, Probability, Sampler

__version__ = SDK_VERSION
log = logging.getLogger("inferlog")

_atexit_registered = False


def _validate_init_args(endpoint: str | None, auth_scheme: str) -> None:
    if endpoint is not None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"inferlog.init: endpoint must be an http(s) URL, got {endpoint!r}"
            )
        if not parsed.netloc:
            raise ValueError(
                f"inferlog.init: endpoint {endpoint!r} is missing a host"
            )
    if auth_scheme not in ("x-api-key", "bearer"):
        raise ValueError(
            f"inferlog.init: auth_scheme must be 'x-api-key' or 'bearer', "
            f"got {auth_scheme!r}"
        )


def init(
    *,
    service: str = "app",
    endpoint: str | None = None,
    api_key: str | None = None,
    enabled: bool = True,
    redactor: Redactor | None = None,
    sampler: Sampler | None = None,
    on_drop: Callable[[int, str], None] | None = None,
    auth_scheme: str = "x-api-key",
    sink: Any = None,
    dispatcher_options: dict | None = None,
    instrument: bool = True,
    register_atexit: bool = True,
) -> list[str]:
    """Initialise InferLog. Call once at process startup.

    Parameters
    ----------
    service:
        Logical name of the emitting application (e.g. ``chat-gateway``).
    endpoint:
        URL of the ingestion API. If None and ``sink`` is also None, events
        are routed to a NullSink (no-op) — useful for tests / dry-run.
    api_key:
        Shared secret presented on every ingest request.
    auth_scheme:
        ``"x-api-key"`` (default) or ``"bearer"``. Picks the HTTP header.
    enabled:
        If False, init becomes a no-op; useful in CI or when feature-flagged.
    redactor:
        Custom :class:`Redactor`. Default redacts emails, phones, cards,
        SSNs, IPs, and API-key-shaped strings.
    sampler:
        Custom :class:`Sampler`. Default is :class:`KeepAll`. Wrap in
        :class:`AlwaysKeepErrors` to guarantee errors are never sampled out.
    on_drop:
        Callback ``(count, reason)`` invoked when events are dropped — for
        host-side observability. Reasons: ``"queue_full"``,
        ``"max_retries"``, ``"permanent_error"``, ``"sampled"``.
    sink:
        Advanced: provide your own Sink. Overrides ``endpoint``.
    dispatcher_options:
        Forwarded to :class:`LogDispatcher` (max_queue, batch_size,
        flush_interval, max_retries).
    instrument:
        If True (default), auto-instrument ``openai`` and ``anthropic``
        client libraries that are importable in this process.
    register_atexit:
        If True (default), register an ``atexit`` hook to flush remaining
        events before process exit. Best-effort.

    Returns
    -------
    list[str]
        The provider libraries actually auto-instrumented.

    Raises
    ------
    ValueError
        If ``endpoint`` is set but not a valid http(s) URL, or
        ``auth_scheme`` is unknown.
    """
    _validate_init_args(endpoint, auth_scheme)

    existing = get_runtime()
    if existing is not None:
        log.warning(
            "inferlog.init called twice; replacing the active runtime. "
            "Call inferlog.shutdown() first to silence this warning.",
        )

    rt = build_default_runtime(
        service=service,
        endpoint=endpoint,
        api_key=api_key,
        enabled=enabled,
        redactor=redactor,
        sink=sink,
        sampler=sampler,
        auth_scheme=auth_scheme,
        dispatcher_options={"on_drop": on_drop, **(dispatcher_options or {})},
    )
    rt.dispatcher.start()  # idempotent; lazy if there's no running loop
    set_runtime(rt)

    if register_atexit:
        _ensure_atexit_handler()

    log.info(
        "inferlog initialised (service=%s, enabled=%s, endpoint=%s)",
        service, enabled, endpoint or "<null>",
    )

    if not instrument or not enabled:
        return []
    return _auto.install(rt)


def _ensure_atexit_handler() -> None:
    """Register the atexit flush once per process."""
    global _atexit_registered
    if _atexit_registered:
        return
    _atexit_registered = True
    atexit.register(_atexit_flush)


def _atexit_flush() -> None:
    """Best-effort drain on interpreter shutdown. Sync only — atexit
    callbacks run after the event loop, so we cannot await."""
    rt = get_runtime()
    if rt is None:
        return
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(rt.dispatcher.aclose(drain_timeout=2.0))
        finally:
            loop.close()
    except Exception:  # noqa: BLE001 — never raise from atexit
        log.debug("inferlog atexit flush raised", exc_info=True)


def flush(timeout: float = 3.0) -> None:
    """Best-effort drain of the dispatcher queue. Call before process exit.

    Safe from sync or async contexts. From within a running event loop the
    flush is scheduled and the caller should `await ashutdown()` to be sure
    everything is delivered.
    """
    rt = get_runtime()
    if rt is None:
        return
    try:
        loop = asyncio.get_running_loop()
        # Inside a loop — schedule and return (caller can await ashutdown).
        loop.create_task(rt.dispatcher.aclose(drain_timeout=timeout))
        return
    except RuntimeError:
        pass
    try:
        asyncio.run(rt.dispatcher.aclose(drain_timeout=timeout))
    except RuntimeError:
        pass


def shutdown() -> None:
    """Tear down. Removes auto-instrumentation patches and clears the
    runtime. Best effort — for guaranteed flush, ``await ashutdown()``."""
    rt = get_runtime()
    if rt is None:
        return
    _auto.uninstall()
    rt.enabled = False
    try:
        asyncio.get_running_loop()
        # Inside a loop — the caller is responsible for awaiting drain.
    except RuntimeError:
        try:
            asyncio.run(rt.dispatcher.aclose())
        except RuntimeError:
            pass
    set_runtime(None)


async def ashutdown(drain_timeout: float = 3.0) -> None:
    """Async tear down. Drains the dispatcher cleanly. Use from async code."""
    rt = get_runtime()
    if rt is None:
        return
    _auto.uninstall()
    rt.enabled = False
    await rt.dispatcher.aclose(drain_timeout=drain_timeout)
    set_runtime(None)


def stats() -> dict:
    """Health snapshot — useful for host-app dashboards."""
    rt = get_runtime()
    if rt is None:
        return {"initialised": False}
    return {
        "initialised": True,
        "service": rt.service,
        "enabled": rt.enabled,
        **rt.dispatcher.stats(),
    }


__all__ = [
    # Public API
    "init", "shutdown", "ashutdown", "flush", "stats", "context",
    # Building blocks customers compose
    "Redactor", "Runtime",
    "Sampler", "KeepAll", "Probability", "AlwaysKeepErrors", "CustomSampler",
    # Explicit-wrapper API (legacy / custom providers)
    "LoggedLLMClient", "LogDispatcher",
    "HttpSink", "MemorySink", "NullSink", "TransientDeliveryError",
    "InferenceEvent", "SDK_VERSION",
    "ChatMessage", "Completion", "StreamChunk", "Usage",
    # Low-level helpers
    "get_runtime", "is_initialized", "current_tags",
]
