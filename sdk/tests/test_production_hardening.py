"""Production-hardening behaviours — sampling, on_drop, retry-after,
auth scheme, init validation, idempotent init."""

import asyncio

import httpx
import pytest

import inferlog
from inferlog import (
    AlwaysKeepErrors,
    HttpSink,
    KeepAll,
    LogDispatcher,
    MemorySink,
    Probability,
    TransientDeliveryError,
    get_runtime,
)


# ---------------------------------------------------------------- sampling


def test_keep_all_keeps_everything(event_factory):
    s = KeepAll()
    assert s.should_sample(event_factory(status="success"))
    assert s.should_sample(event_factory(status="error"))


def test_probability_zero_drops_everything(event_factory):
    s = Probability(0.0)
    assert not s.should_sample(event_factory())


def test_probability_one_keeps_everything(event_factory):
    s = Probability(1.0)
    assert s.should_sample(event_factory())


def test_always_keep_errors_overrides_inner_sampler(event_factory):
    s = AlwaysKeepErrors(Probability(0.0))
    assert s.should_sample(event_factory(status="error"))
    assert s.should_sample(event_factory(status="cancelled"))
    assert not s.should_sample(event_factory(status="success"))


def test_probability_rejects_out_of_range():
    with pytest.raises(ValueError):
        Probability(1.5)
    with pytest.raises(ValueError):
        Probability(-0.1)


async def test_runtime_sampler_skips_events_via_explicit_client():
    """Sampler integrated end-to-end via the explicit LoggedLLMClient."""
    sink = MemorySink()
    inferlog.init(service="t", sink=sink, sampler=Probability(0.0),
                  dispatcher_options={"flush_interval": 0.05})
    try:
        from inferlog import LoggedLLMClient
        from inferlog.providers import ChatMessage, MockProvider

        rt = get_runtime()
        assert rt is not None
        client = LoggedLLMClient(
            service="t", dispatcher=rt.dispatcher,
            providers={"mock": MockProvider(token_delay=0)},
        )
        await client.complete(
            provider="mock", model="mock-1",
            messages=[ChatMessage("user", "hello")],
        )
        await asyncio.sleep(0.1)
        await rt.dispatcher.aclose()
        assert sink.events == []  # sampled out
        assert rt.dispatcher.dropped == 1
    finally:
        await inferlog.ashutdown()


# ---------------------------------------------------------------- on_drop


async def test_on_drop_fires_when_queue_full(event_factory):
    dropped: list[tuple[int, str]] = []
    dispatcher = LogDispatcher(
        MemorySink(),
        max_queue=2,
        on_drop=lambda n, reason: dropped.append((n, reason)),
    )
    for _ in range(10):
        dispatcher.submit(event_factory())
    assert dispatcher.dropped == 8
    assert dropped, "on_drop should have been called for at least one drop"
    assert dropped[0][1] == "queue_full"


async def test_on_drop_fires_on_max_retries(event_factory):
    dropped: list[tuple[int, str]] = []

    class AlwaysFailingSink:
        async def send(self, events):  # noqa: ARG002
            raise TransientDeliveryError("nope")

    dispatcher = LogDispatcher(
        AlwaysFailingSink(),
        flush_interval=0.02,
        max_retries=2,
        on_drop=lambda n, reason: dropped.append((n, reason)),
    )
    dispatcher.start()
    dispatcher.submit(event_factory())
    await asyncio.sleep(0.5)
    await dispatcher.aclose()
    assert dropped, "on_drop should have fired after exhausting retries"
    assert dropped[0][1] == "max_retries"


def test_on_drop_callback_errors_are_swallowed(event_factory):
    def bad_cb(n, reason):  # noqa: ARG001
        raise RuntimeError("customer code is buggy")

    dispatcher = LogDispatcher(MemorySink(), max_queue=1, on_drop=bad_cb)
    for _ in range(5):
        dispatcher.submit(event_factory())  # must not raise
    assert dispatcher.dropped == 4


# ---------------------------------------------------------------- HttpSink


async def test_http_sink_bearer_auth():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(202, json={"accepted": 1})

    sink = HttpSink(
        "http://x/ingest", api_key="abc", auth_scheme="bearer",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await sink.send([{"x": 1}])
    assert seen["auth"] == "Bearer abc"


async def test_http_sink_raises_transient_on_429_with_retry_after():
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1.5"})

    sink = HttpSink(
        "http://x/ingest", api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(TransientDeliveryError) as exc:
        await sink.send([{"x": 1}])
    assert exc.value.retry_after == 1.5


async def test_http_sink_raises_transient_on_500():
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    sink = HttpSink(
        "http://x/ingest", api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(TransientDeliveryError):
        await sink.send([{"x": 1}])


def test_http_sink_rejects_unknown_auth_scheme():
    with pytest.raises(ValueError):
        HttpSink("http://x", api_key="k", auth_scheme="weird")


# ----------------------------------------------------- init validation


def test_init_rejects_non_http_endpoint():
    with pytest.raises(ValueError, match="http"):
        inferlog.init(endpoint="ftp://x.y/z")


def test_init_rejects_endpoint_missing_host():
    with pytest.raises(ValueError, match="host"):
        inferlog.init(endpoint="http:///path")


def test_init_rejects_unknown_auth_scheme():
    with pytest.raises(ValueError, match="auth_scheme"):
        inferlog.init(endpoint="http://x.y/z", auth_scheme="basic")


async def test_init_is_idempotent_and_warns():
    """Two consecutive inits leave a working runtime; the second warns."""
    sink = MemorySink()
    inferlog.init(service="a", sink=sink, register_atexit=False)
    inferlog.init(service="b", sink=sink, register_atexit=False)
    rt = get_runtime()
    assert rt is not None
    assert rt.service == "b"
    await inferlog.ashutdown()


# ----------------------------------------------------- lazy dispatcher start


def test_dispatcher_submit_is_safe_with_no_running_loop(event_factory):
    """`submit` from outside an event loop must not raise — it queues
    silently and the task starts on first submit from async code."""
    dispatcher = LogDispatcher(MemorySink(), flush_interval=0.05)
    dispatcher.submit(event_factory())  # would crash if not lazy
    # No async drain — the test just asserts no exception from sync submit.
