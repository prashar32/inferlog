import pytest

from inferlog.client import LoggedLLMClient
from inferlog.providers import ChatMessage, MockProvider, Provider, StreamChunk


class CollectingDispatcher:
    """Stand-in for LogDispatcher that just records emitted events."""

    def __init__(self):
        self.events = []

    def submit(self, event):
        self.events.append(event)


class RateLimitError(Exception):
    """Named so the client's error classifier maps it to 'rate_limit'."""


class BoomProvider(Provider):
    name = "boom"

    async def complete(self, model, messages, **opts):
        raise RateLimitError("slow down")

    async def stream(self, model, messages, **opts):
        yield StreamChunk(text="partial ")
        raise RateLimitError("slow down")


def make_client(provider: Provider, name: str) -> tuple[LoggedLLMClient, CollectingDispatcher]:
    dispatcher = CollectingDispatcher()
    client = LoggedLLMClient(
        service="test-gateway", dispatcher=dispatcher, providers={name: provider}
    )
    return client, dispatcher


async def test_complete_emits_success_event():
    client, dispatcher = make_client(MockProvider(token_delay=0), "mock")
    result = await client.complete(
        provider="mock",
        model="mock-1",
        messages=[ChatMessage("user", "hello")],
        conversation_id="conv-1",
    )
    assert result.text
    (event,) = dispatcher.events
    assert event.status == "success"
    assert event.streamed is False
    assert event.conversation_id == "conv-1"
    assert event.total_tokens and event.total_tokens > 0
    assert event.input_preview == "hello"
    assert event.output_preview


async def test_stream_emits_event_with_ttft():
    client, dispatcher = make_client(MockProvider(token_delay=0), "mock")
    text = ""
    async for chunk in client.stream(
        provider="mock", model="mock-1", messages=[ChatMessage("user", "hi")]
    ):
        text += chunk.text
    assert text
    (event,) = dispatcher.events
    assert event.status == "success"
    assert event.streamed is True
    assert event.ttft_ms is not None and event.ttft_ms >= 0
    assert event.output_preview


async def test_complete_error_is_classified_and_reraised():
    client, dispatcher = make_client(BoomProvider(), "boom")
    with pytest.raises(RateLimitError):
        await client.complete(
            provider="boom", model="x", messages=[ChatMessage("user", "hi")]
        )
    (event,) = dispatcher.events
    assert event.status == "error"
    assert event.error_type == "rate_limit"
    assert event.error_message


async def test_cancelled_stream_is_logged_as_cancelled():
    client, dispatcher = make_client(MockProvider(token_delay=0.01), "mock")
    stream = client.stream(
        provider="mock", model="mock-1", messages=[ChatMessage("user", "hello")]
    )
    # consume one chunk, then close the stream early — like the user hitting Stop.
    await stream.__anext__()
    await stream.aclose()

    (event,) = dispatcher.events
    assert event.status == "cancelled"
    assert event.streamed is True
    # whatever was generated before cancel is still captured
    assert event.output_preview


async def test_unknown_provider_raises():
    client, _ = make_client(MockProvider(), "mock")
    with pytest.raises(ValueError, match="unknown provider"):
        await client.complete(
            provider="nope", model="x", messages=[ChatMessage("user", "hi")]
        )
