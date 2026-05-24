"""inferlog.wrap_provider — the supported entry point for non-HTTP providers.

Customers should never reach into Runtime internals to construct a
LoggedLLMClient by hand. This test pins that contract.
"""

from __future__ import annotations

import pytest

import inferlog
from inferlog.providers import ChatMessage, MockProvider


@pytest.fixture(autouse=True)
async def fresh_runtime():
    inferlog.shutdown()
    yield
    await inferlog.ashutdown()


async def test_wrap_provider_requires_init():
    """Calling wrap_provider before init() must fail loudly, not silently."""
    with pytest.raises(RuntimeError, match="inferlog.init"):
        inferlog.wrap_provider(mock=MockProvider(token_delay=0))


async def test_wrap_provider_rejects_empty():
    inferlog.init(service="t", capture_all_httpx=False)
    with pytest.raises(ValueError, match="at least one provider"):
        inferlog.wrap_provider()


async def test_wrap_provider_emits_events_through_runtime_dispatcher():
    """A call through the wrapped client must land on the same dispatcher
    init() configured — that's the whole reason wrap_provider exists."""
    sink = inferlog.MemorySink()
    inferlog.init(service="t", sink=sink, capture_all_httpx=False)
    client = inferlog.wrap_provider(mock=MockProvider(token_delay=0))

    result = await client.complete(
        provider="mock", model="m-1",
        messages=[ChatMessage("user", "hello")],
    )
    assert result.text

    await inferlog.ashutdown()
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["provider"] == "mock"
    assert event["model"] == "m-1"
    assert event["status"] == "success"


async def test_wrap_provider_carries_context_tags():
    sink = inferlog.MemorySink()
    inferlog.init(service="t", sink=sink, capture_all_httpx=False)
    client = inferlog.wrap_provider(mock=MockProvider(token_delay=0))

    with inferlog.context(conversation_id="abc-123", user_id="u-1"):
        await client.complete(
            provider="mock", model="m-1",
            messages=[ChatMessage("user", "hi")],
        )

    await inferlog.ashutdown()
    event = sink.events[0]
    assert event["conversation_id"] == "abc-123"
    assert event["tags"]["user_id"] == "u-1"
