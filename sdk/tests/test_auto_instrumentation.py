"""Auto-instrumentation — the customer calls `inferlog.init()` once and
their existing openai code is captured. These tests assert that contract.

We mock the HTTP layer with `httpx.MockTransport`, so the real openai
client + our real monkey-patch run end-to-end without any network.
"""

import asyncio
import json

import httpx
import pytest

import inferlog
from inferlog import MemorySink, get_runtime


def _openai_completion(content: str, prompt_tokens: int = 10, completion_tokens: int = 2):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_openai_with_mock(handler):
    from openai import AsyncOpenAI
    transport = httpx.MockTransport(handler)
    return AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))


@pytest.fixture
async def sink_and_init():
    """Initialise the SDK against a MemorySink, yield it, tear down."""
    sink = MemorySink()
    installed = inferlog.init(service="auto-test", sink=sink,
                              dispatcher_options={"flush_interval": 0.05})
    yield sink, installed
    # tear down so other tests aren't polluted
    await inferlog.ashutdown()


async def test_init_reports_installed_providers(sink_and_init):
    _, installed = sink_and_init
    # openai and anthropic are installed in this venv
    assert "openai" in installed


async def test_openai_non_streaming_is_captured(sink_and_init):
    sink, _ = sink_and_init

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_completion("Hi there!"))

    client = _make_openai_with_mock(handler)
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "reach me at a@b.com"}],
    )

    assert resp.choices[0].message.content == "Hi there!"

    # let the dispatcher flush
    await asyncio.sleep(0.15)
    rt = get_runtime()
    assert rt is not None
    await rt.dispatcher.aclose()

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["provider"] == "openai"
    assert event["model"] == "gpt-4o-mini"
    assert event["status"] == "success"
    assert event["streamed"] is False
    assert event["prompt_tokens"] == 10
    assert event["completion_tokens"] == 2
    # PII redacted before the event reached the sink
    assert "a@b.com" not in (event["input_preview"] or "")
    assert event["pii_redaction_count"] >= 1
    # The auto path marks itself in client_metadata
    assert event["client_metadata"].get("auto_instrumented") is True


async def test_openai_error_is_captured(sink_and_init):
    sink, _ = sink_and_init

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
        )

    client = _make_openai_with_mock(handler)
    with pytest.raises(Exception):
        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    await asyncio.sleep(0.15)
    rt = get_runtime()
    assert rt is not None
    await rt.dispatcher.aclose()

    error_events = [e for e in sink.events if e["status"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["error_type"] == "rate_limit"


async def test_context_tags_propagate_to_events(sink_and_init):
    sink, _ = sink_and_init

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_completion("ok"))

    client = _make_openai_with_mock(handler)
    with inferlog.context(conversation_id="conv-xyz", user_id="user-123"):
        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
        )

    await asyncio.sleep(0.15)
    rt = get_runtime()
    assert rt is not None
    await rt.dispatcher.aclose()

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["conversation_id"] == "conv-xyz"
    assert event["tags"].get("user_id") == "user-123"


async def test_openai_streaming_is_captured_with_ttft(sink_and_init):
    """Streaming response: assemble a fake SSE body and assert TTFT and
    usage make it into the event."""
    sink, _ = sink_and_init

    chunks = [
        # text chunks
        {"id": "1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
         "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
        {"id": "2", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
         "choices": [{"index": 0, "delta": {"content": "lo!"}, "finish_reason": None}]},
        # final usage chunk
        {"id": "3", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
         "choices": [],
         "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_openai_with_mock(handler)
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "say hello"}],
        stream=True,
    )
    received = []
    async for chunk in stream:
        received.append(chunk)
    assert any(c.choices and c.choices[0].delta.content for c in received)

    await asyncio.sleep(0.15)
    rt = get_runtime()
    assert rt is not None
    await rt.dispatcher.aclose()

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["streamed"] is True
    assert event["status"] == "success"
    assert event["ttft_ms"] is not None and event["ttft_ms"] >= 0
    assert event["prompt_tokens"] == 8
    assert event["completion_tokens"] == 3
