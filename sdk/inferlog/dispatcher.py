"""Background delivery of inference events.

Design goal: logging must never slow down or break the host request path.
So `submit()` is sync, non-blocking, and drops events under pressure
rather than raising. Actual HTTP delivery happens on a background task
that batches events and retries with backoff + jitter, honouring
`Retry-After` on 429/503.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Callable, Protocol

import httpx

from .events import InferenceEvent

log = logging.getLogger("inferlog.dispatcher")


class TransientDeliveryError(Exception):
    """Raised by a Sink to ask the dispatcher to retry after `retry_after`
    seconds (e.g. on HTTP 429 / 503 with Retry-After). When `retry_after`
    is None, the dispatcher's normal exponential backoff applies."""

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class Sink(Protocol):
    """Where a batch of event payloads ultimately goes."""

    async def send(self, events: list[dict]) -> None: ...


class HttpSink:
    """Posts batches to the ingestion API as `{"events": [...]}`.

    Configurable auth scheme:
      * `auth_scheme="x-api-key"` (default) — sends `x-api-key: <api_key>`.
      * `auth_scheme="bearer"` — sends `Authorization: Bearer <api_key>`.

    On HTTP 429 / 503 the sink raises `TransientDeliveryError` with the
    `Retry-After` value parsed out, so the dispatcher waits the right
    amount of time before retrying.
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        auth_scheme: str = "x-api-key",
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
        extra_headers: dict | None = None,
    ):
        self._url = url
        headers: dict[str, str] = {}
        if api_key:
            if auth_scheme == "x-api-key":
                headers["x-api-key"] = api_key
            elif auth_scheme == "bearer":
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                raise ValueError(
                    f"unknown auth_scheme {auth_scheme!r}; expected 'x-api-key' or 'bearer'"
                )
        if extra_headers:
            headers.update(extra_headers)
        self._headers = headers
        # An explicit client lets the caller share a connection pool or
        # inject a mock transport in tests.
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def send(self, events: list[dict]) -> None:
        try:
            resp = await self._client.post(
                self._url, json={"events": events}, headers=self._headers
            )
        except httpx.TransportError as exc:
            # Network-level failure — definitely worth retrying.
            raise TransientDeliveryError(f"transport error: {exc}") from exc

        if resp.status_code in (429, 503):
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            raise TransientDeliveryError(
                f"server returned {resp.status_code}", retry_after=retry_after
            )
        if 500 <= resp.status_code < 600:
            raise TransientDeliveryError(f"server returned {resp.status_code}")
        resp.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_retry_after(header: str | None) -> float | None:
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        # HTTP-date form. We don't parse it precisely; just back off ~30s.
        return 30.0


class MemorySink:
    """Collects events in memory. Used by tests and offline runs."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def send(self, events: list[dict]) -> None:
        self.events.extend(events)


class NullSink:
    """Discards everything. Handy to disable logging without code changes."""

    async def send(self, events: list[dict]) -> None:  # noqa: D401
        return None


class LogDispatcher:
    """Bounded queue + background flusher in front of a Sink.

    "Near real time" here means: events are flushed either when a batch
    fills up or after `flush_interval` seconds, whichever comes first.

    Properties production users care about:
      * **Lazy start.** `submit()` is safe before an event loop exists;
        the background task starts on first submit from an async context.
      * **Drops are visible.** `on_drop(event_count, reason)` callback,
        plus the `dropped` counter on `stats()`.
      * **Polite retry.** Exponential backoff with jitter; honours the
        `Retry-After` header on 429 / 503.
      * **Idempotent close.** Safe to call `aclose()` multiple times.
    """

    def __init__(
        self,
        sink: Sink,
        *,
        max_queue: int = 2000,
        batch_size: int = 25,
        flush_interval: float = 0.5,
        max_retries: int = 3,
        on_drop: Callable[[int, str], None] | None = None,
    ):
        self._sink = sink
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._on_drop = on_drop
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self._closing = False
        self._closed = False
        # Observability for the logger itself — exposed via stats().
        self.dropped = 0
        self.delivered = 0

    def start(self) -> None:
        """Eagerly start the background task. Idempotent; safe to call from
        any async context. From a sync context with no running loop this is
        a no-op — the task will start on first `submit()` from async code.
        """
        if self._task is not None or self._closed:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop; defer to first submit() from async code.
            return
        self._task = asyncio.create_task(self._run(), name="inferlog-dispatcher")

    def submit(self, event: InferenceEvent) -> None:
        """Hand an event to the dispatcher. Never blocks, never raises."""
        if self._closed:
            return
        # Lazy start — covers the case where init() was called from a sync
        # context and the loop only started later.
        if self._task is None:
            self.start()
        try:
            self._queue.put_nowait(event.to_payload())
        except asyncio.QueueFull:
            # Backpressure: shed load instead of stalling the request path.
            self.dropped += 1
            self._notify_drop(1, "queue_full")
            log.warning("inferlog queue full — dropped event %s", event.request_id)

    def stats(self) -> dict:
        return {
            "queued": self._queue.qsize(),
            "delivered": self.delivered,
            "dropped": self.dropped,
            "closed": self._closed,
        }

    async def aclose(self, drain_timeout: float = 3.0) -> None:
        """Flush what we can, then stop the background task. Idempotent."""
        if self._closed:
            return
        self._closing = True
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=drain_timeout)
            except asyncio.TimeoutError:
                self._task.cancel()
        if isinstance(self._sink, HttpSink):
            await self._sink.aclose()
        self._closed = True

    def _notify_drop(self, count: int, reason: str) -> None:
        if self._on_drop is None:
            return
        try:
            self._on_drop(count, reason)
        except Exception:  # noqa: BLE001 — customer callback must not break us
            log.debug("inferlog on_drop callback raised", exc_info=True)

    async def _run(self) -> None:
        while not (self._closing and self._queue.empty()):
            batch = await self._collect_batch()
            if batch:
                await self._send_with_retry(batch)

    async def _collect_batch(self) -> list[dict]:
        """Wait for at least one event, then drain up to batch_size."""
        batch: list[dict] = []
        try:
            first = await asyncio.wait_for(
                self._queue.get(), timeout=self._flush_interval
            )
            batch.append(first)
        except asyncio.TimeoutError:
            return batch
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _send_with_retry(self, batch: list[dict]) -> None:
        delay = 0.25
        for attempt in range(1, self._max_retries + 1):
            try:
                await self._sink.send(batch)
                self.delivered += len(batch)
                return
            except TransientDeliveryError as exc:
                if attempt == self._max_retries:
                    self.dropped += len(batch)
                    self._notify_drop(len(batch), "max_retries")
                    log.error(
                        "inferlog gave up on %d events after %d retries: %s",
                        len(batch), attempt, exc,
                    )
                    return
                # Honour Retry-After if present; otherwise jittered backoff.
                sleep_for = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else delay * (0.5 + random.random())
                )
                await asyncio.sleep(sleep_for)
                delay *= 2
            except Exception as exc:  # noqa: BLE001 — never crash the loop
                if attempt == self._max_retries:
                    self.dropped += len(batch)
                    self._notify_drop(len(batch), "permanent_error")
                    log.error(
                        "inferlog dropped %d events (permanent error): %s",
                        len(batch), exc,
                    )
                    return
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay *= 2
