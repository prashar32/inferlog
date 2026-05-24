"""Transport mode — surgical capture without globally patching httpx.

Proves the contract: with `capture_all_httpx=False`, our code does NOT
sit in the path of any httpx call EXCEPT those made through a client
the customer explicitly attached `inferlog.transport()` to.

This is the right enterprise default: zero coupling with unrelated
httpx traffic; customer opts in per-client.
"""

import asyncio
import json

import httpx
import pytest_asyncio

import inferlog
from inferlog import MemorySink, get_runtime


@pytest_asyncio.fixture
async def sink_no_global():
    """Init inferlog WITHOUT globally patching httpx."""
    sink = MemorySink()
    installed = inferlog.init(
        service="transport-test",
        sink=sink,
        capture_all_httpx=False,  # the key flag
        dispatcher_options={"flush_interval": 0.05},
        register_atexit=False,
    )
    assert installed == [], "init(capture_all_httpx=False) must not patch httpx globally"
    yield sink
    await inferlog.ashutdown()


async def _drain():
    await asyncio.sleep(0.15)
    rt = get_runtime()
    assert rt is not None
    await rt.dispatcher.aclose()


# -- positive cases — the transport DOES capture --------------------------


async def test_transport_captures_openai_non_streaming(sink_no_global):
    sink = sink_no_global

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    # Customer attaches our transport on top of their mock transport.
    wrapped = inferlog.transport(inner=httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=wrapped) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "ping with a@b.com"}],
            },
        )
        assert resp.status_code == 200

    await _drain()
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["provider"] == "openai"
    assert event["model"] == "gpt-4o-mini"
    assert event["prompt_tokens"] == 5
    # PII still redacted in-process
    assert "a@b.com" not in (event["input_preview"] or "")


async def test_transport_captures_anthropic_streaming(sink_no_global):
    sink = sink_no_global

    events = [
        {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 4}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "message_delta", "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    body = b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    wrapped = inferlog.transport(inner=httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=wrapped) as client:
        async with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            json={"model": "claude-3-haiku",
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 10, "stream": True},
        ) as resp:
            async for _ in resp.aiter_bytes():
                pass

    await _drain()
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["provider"] == "anthropic"
    assert event["streamed"] is True
    assert event["prompt_tokens"] == 4
    assert event["completion_tokens"] == 1


# -- the headline test — NO coupling on a different client ----------------


async def test_unrelated_httpx_client_is_NOT_in_our_path(sink_no_global):
    """The point of the transport mode: a different httpx client (e.g.,
    the customer's database HTTP client, Stripe call, internal RPC)
    runs through stock httpx with zero inferlog code in the way."""

    seen_unrelated = []

    def stripe_handler(req: httpx.Request) -> httpx.Response:
        seen_unrelated.append(req.url.path)
        return httpx.Response(200, json={"id": "ch_test", "amount": 1000})

    # NO inferlog transport on this client — vanilla httpx.
    async with httpx.AsyncClient(transport=httpx.MockTransport(stripe_handler)) as client:
        resp = await client.post(
            "https://api.stripe.com/v1/charges", json={"amount": 1000},
        )
        assert resp.status_code == 200

    await _drain()
    # The unrelated call happened, and we emitted NOTHING — we're not in
    # the path of that client at all.
    assert seen_unrelated == ["/v1/charges"]
    assert sink_no_global.events == []


async def test_two_clients_one_instrumented_one_not(sink_no_global):
    """Mix the two patterns in the same process: one client uses our
    transport (instrumented), another doesn't (untouched). Only the
    LLM client's events land in the sink."""
    sink = sink_no_global

    def openai_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        })

    def random_api_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"weather": "sunny"})

    instrumented = inferlog.transport(inner=httpx.MockTransport(openai_handler))
    plain = httpx.MockTransport(random_api_handler)

    async with httpx.AsyncClient(transport=instrumented) as oai_client, \
            httpx.AsyncClient(transport=plain) as other_client:
        # LLM call — captured.
        r1 = await oai_client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o-mini",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        # Unrelated API call — untouched.
        r2 = await other_client.get("https://weather.example.com/today")
        assert r1.status_code == 200
        assert r2.status_code == 200

    await _drain()
    assert len(sink.events) == 1
    assert sink.events[0]["provider"] == "openai"


# -- transport defers if global patch is already active -------------------


async def test_transport_defers_when_global_patch_is_active():
    """If the customer (or an upstream library) called `init()` with the
    default `capture_all_httpx=True`, the transport steps aside so we
    don't double-log."""
    sink = MemorySink()
    inferlog.init(
        service="dual-mode-test",
        sink=sink,
        capture_all_httpx=True,  # the global default
        dispatcher_options={"flush_interval": 0.05},
        register_atexit=False,
    )
    try:
        def openai_handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            })

        wrapped = inferlog.transport(inner=httpx.MockTransport(openai_handler))
        async with httpx.AsyncClient(transport=wrapped) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini",
                      "messages": [{"role": "user", "content": "hi"}]},
            )

        await asyncio.sleep(0.15)
        rt = get_runtime()
        assert rt is not None
        await rt.dispatcher.aclose()

        # Exactly one event — the global patch captured, the transport stood aside.
        assert len(sink.events) == 1
    finally:
        await inferlog.ashutdown()
