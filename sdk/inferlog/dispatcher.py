"""Background delivery of inference events.

Design goal: logging must never slow down or break a chat response. So
`submit()` is sync, non-blocking, and drops events under pressure rather
than raising. Actual HTTP delivery happens on a background task that
batches events and retries with backoff.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import httpx

from .events import InferenceEvent

log = logging.getLogger("inferlog.dispatcher")


class Sink(Protocol):
    """Where a batch of event payloads ultimately goes."""

    async def send(self, events: list[dict]) -> None: ...


class HttpSink:
    """Posts batches to the ingestion API as `{"events": [...]}`."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._url = url
        self._headers = {"x-api-key": api_key} if api_key else {}
        # An explicit client can be passed to share a connection pool (or to
        # inject a mock transport in tests).
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def send(self, events: list[dict]) -> None:
        resp = await self._client.post(
            self._url, json={"events": events}, headers=self._headers
        )
        resp.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()


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

    `near real time` here means: events are flushed either when a batch
    fills up or after `flush_interval` seconds, whichever comes first.
    """

    def __init__(
        self,
        sink: Sink,
        *,
        max_queue: int = 2000,
        batch_size: int = 25,
        flush_interval: float = 0.5,
        max_retries: int = 3,
    ):
        self._sink = sink
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self._closing = False
        # Observability for the logger itself — exposed via stats().
        self.dropped = 0
        self.delivered = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="inferlog-dispatcher")

    def submit(self, event: InferenceEvent) -> None:
        """Hand an event to the dispatcher. Never blocks, never raises."""
        try:
            self._queue.put_nowait(event.to_payload())
        except asyncio.QueueFull:
            # Backpressure: shed load instead of stalling the request path.
            self.dropped += 1
            log.warning("inferlog queue full — dropped event %s", event.request_id)

    def stats(self) -> dict:
        return {
            "queued": self._queue.qsize(),
            "delivered": self.delivered,
            "dropped": self.dropped,
        }

    async def aclose(self, drain_timeout: float = 3.0) -> None:
        """Flush what we can, then stop the background task."""
        self._closing = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=drain_timeout)
            except asyncio.TimeoutError:
                self._task.cancel()
        if isinstance(self._sink, HttpSink):
            await self._sink.aclose()

    async def _run(self) -> None:
        while not (self._closing and self._queue.empty()):
            batch = await self._collect_batch()
            if batch:
                await self._send_with_retry(batch)

    async def _collect_batch(self) -> list[dict]:
        """Wait for at least one event, then drain up to batch_size."""
        batch: list[dict] = []
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
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
            except Exception as exc:  # noqa: BLE001 — delivery is best-effort
                if attempt == self._max_retries:
                    self.dropped += len(batch)
                    log.error("inferlog gave up on %d events: %s", len(batch), exc)
                    return
                await asyncio.sleep(delay)
                delay *= 2
