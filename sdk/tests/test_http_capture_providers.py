"""Proof of model-agnosticism — capture works for multiple providers
and even for a raw httpx call with no vendor SDK at all.

We never write per-call code. Just `inferlog.init(...)` once, and the
HTTP layer does the rest.
"""

import asyncio
import json

import httpx
import pytest_asyncio

import inferlog
from inferlog import MemorySink, get_runtime


@pytest_asyncio.fixture
async def sink_and_init():
    sink = MemorySink()
    inferlog.init(
        service="http-capture-test",
        sink=sink,
        dispatcher_options={"flush_interval": 0.05},
        register_atexit=False,
    )
    yield sink
    await inferlog.ashutdown()


async def _drain():
    await asyncio.sleep(0.15)
    rt = get_runtime()
    assert rt is not None
    await rt.dispatcher.aclose()


# ============================================================ Anthropic


async def test_anthropic_non_streaming_via_httpx(sink_and_init):
    """A raw httpx POST to api.anthropic.com is captured — we never even
    use the anthropic SDK in this test."""
    sink = sink_and_init

    def handler(req: httpx.Request) -> httpx.Response:
        assert "anthropic" in req.url.host
        return httpx.Response(200, json={
            "id": "msg_test", "type": "message", "model": "claude-3-5-sonnet",
            "content": [{"type": "text", "text": "Hello from Claude"}],
            "usage": {"input_tokens": 12, "output_tokens": 4},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "ping with a@b.com"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200

    await _drain()
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["provider"] == "anthropic"
    assert event["model"] == "claude-3-5-sonnet-latest"
    assert event["status"] == "success"
    assert event["prompt_tokens"] == 12
    assert event["completion_tokens"] == 4
    assert event["total_tokens"] == 16
    # PII still redacted in-process
    assert "a@b.com" not in (event["input_preview"] or "")
    assert event["pii_redaction_count"] >= 1


async def test_anthropic_streaming_via_httpx(sink_and_init):
    """Anthropic streaming SSE — event-typed protocol — captured."""
    sink = sink_and_init

    events_to_send = [
        {"type": "message_start", "message": {"id": "msg_x", "usage": {"input_tokens": 9}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " there"}},
        {"type": "message_delta", "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]
    body = b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events_to_send
    )

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            json={
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "say hi"}],
                "max_tokens": 100,
                "stream": True,
            },
        ) as resp:
            async for _ in resp.aiter_bytes():
                pass

    await _drain()
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["provider"] == "anthropic"
    assert event["streamed"] is True
    assert event["status"] == "success"
    assert event["prompt_tokens"] == 9
    assert event["completion_tokens"] == 3
    assert event["total_tokens"] == 12


# ============================================================ Ollama


async def test_ollama_non_streaming(sink_and_init):
    """A self-hosted Ollama server — different URL, different shape,
    still captured."""
    sink = sink_and_init

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "llama3",
            "message": {"role": "assistant", "content": "I'm a llama"},
            "done": True,
            "prompt_eval_count": 7,
            "eval_count": 4,
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": [{"role": "user", "content": "hello self-hosted"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200

    await _drain()
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["provider"] == "ollama"
    assert event["model"] == "llama3"
    assert event["status"] == "success"
    assert event["prompt_tokens"] == 7
    assert event["completion_tokens"] == 4
    assert event["total_tokens"] == 11


async def test_ollama_streaming_ndjson(sink_and_init):
    """Ollama streams as newline-delimited JSON, not SSE."""
    sink = sink_and_init

    chunks = [
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo"}, "done": False},
        {"message": {"content": "!"}, "done": False},
        {"done": True, "prompt_eval_count": 5, "eval_count": 3},
    ]
    body = b"".join(f"{json.dumps(c)}\n".encode() for c in chunks)

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with client.stream(
            "POST", "http://localhost:11434/api/chat",
            json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as resp:
            async for _ in resp.aiter_bytes():
                pass

    await _drain()
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["provider"] == "ollama"
    assert event["streamed"] is True
    assert event["status"] == "success"
    assert event["prompt_tokens"] == 5
    assert event["completion_tokens"] == 3


# ============================================================ OpenAI-compatible


async def test_openai_compatible_via_vllm_style_url(sink_and_init):
    """A self-hosted vLLM (or OpenRouter / Together / LiteLLM-proxy)
    speaking the OpenAI Chat Completions wire format. Same parser as
    OpenAI — labelled 'openai_compatible' so you can slice by host."""
    sink = sink_and_init

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "chatcmpl-vllm-1", "object": "chat.completion",
            "created": 1, "model": "mistral-7b-instruct",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "vLLM here"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await client.post(
            "http://my-vllm-server:8000/v1/chat/completions",
            json={
                "model": "mistral-7b-instruct",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert resp.status_code == 200

    await _drain()
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["provider"] == "openai_compatible"
    assert event["model"] == "mistral-7b-instruct"
    assert event["prompt_tokens"] == 6


# ============================================================ Negative cases


async def test_non_llm_post_passes_through_uncaptured(sink_and_init):
    """A POST to an unrelated URL is NOT captured — we only act on URLs
    that match a registered LLM handler."""
    sink = sink_and_init

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await client.post("https://example.com/api/orders", json={"item": 42})
        assert resp.status_code == 200

    await _drain()
    assert sink.events == []


async def test_http_error_is_classified(sink_and_init):
    """5xx / 429 / 4xx come back as `status='error'` with a classified
    `error_type`."""
    sink = sink_and_init

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o-mini",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 429

    await _drain()
    err = [e for e in sink.events if e["status"] == "error"]
    assert len(err) == 1
    assert err[0]["error_type"] == "rate_limit"
    assert err[0]["provider"] == "openai"
