import asyncio

import httpx
import pytest

from inferlog.dispatcher import HttpSink, LogDispatcher, MemorySink


class FlakySink:
    """Fails its first `fail_times` send() calls, then succeeds."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0
        self.events: list[dict] = []

    async def send(self, events):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated ingestion outage")
        self.events.extend(events)


async def _drain(dispatcher: LogDispatcher):
    # give the background task a moment, then close (which flushes).
    await asyncio.sleep(0.05)
    await dispatcher.aclose()


async def test_events_are_delivered_in_batches(event_factory):
    sink = MemorySink()
    dispatcher = LogDispatcher(sink, flush_interval=0.05)
    dispatcher.start()
    for i in range(5):
        dispatcher.submit(event_factory(request_id=f"req-{i}"))
    await _drain(dispatcher)

    assert len(sink.events) == 5
    assert dispatcher.stats()["delivered"] == 5


async def test_delivery_retries_then_succeeds(event_factory):
    sink = FlakySink(fail_times=2)
    dispatcher = LogDispatcher(sink, flush_interval=0.05, max_retries=3)
    dispatcher.start()
    dispatcher.submit(event_factory())
    await _drain(dispatcher)

    assert sink.calls == 3            # two failures + one success
    assert len(sink.events) == 1
    assert dispatcher.dropped == 0


async def test_delivery_gives_up_after_max_retries(event_factory):
    sink = FlakySink(fail_times=99)
    dispatcher = LogDispatcher(sink, flush_interval=0.05, max_retries=2)
    dispatcher.start()
    dispatcher.submit(event_factory())
    await _drain(dispatcher)

    # The event is dropped, but the chat path was never affected.
    assert dispatcher.dropped == 1
    assert sink.events == []


async def test_full_queue_drops_without_raising(event_factory):
    # A tiny queue and no running flusher: submit must never raise.
    dispatcher = LogDispatcher(MemorySink(), max_queue=2)
    for _ in range(10):
        dispatcher.submit(event_factory())
    assert dispatcher.dropped == 8


async def test_http_sink_posts_event_envelope(event_factory):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["body"] = request.read()
        return httpx.Response(202, json={"accepted": 1})

    transport = httpx.MockTransport(handler)
    sink = HttpSink(
        "http://ingestion/v1/ingest",
        api_key="secret",
        client=httpx.AsyncClient(transport=transport),
    )
    dispatcher = LogDispatcher(sink, flush_interval=0.05)
    dispatcher.start()
    dispatcher.submit(event_factory(request_id="abc"))
    await _drain(dispatcher)

    import json

    payload = json.loads(seen["body"])
    assert [e["request_id"] for e in payload["events"]] == ["abc"]
    assert seen["headers"]["x-api-key"] == "secret"
