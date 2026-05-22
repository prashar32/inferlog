import pytest

from inferlog.providers import ChatMessage, MockProvider


async def test_mock_complete_returns_text_and_usage():
    provider = MockProvider(token_delay=0)
    result = await provider.complete(
        "mock-1", [ChatMessage("user", "hello there")]
    )
    assert result.text
    assert result.usage.total_tokens == (
        result.usage.prompt_tokens + result.usage.completion_tokens
    )


async def test_mock_stream_yields_text_then_usage():
    provider = MockProvider(token_delay=0)
    chunks = [
        c async for c in provider.stream("mock-1", [ChatMessage("user", "hi")])
    ]
    text = "".join(c.text for c in chunks)
    assert text
    # the last chunk carries usage, like real providers
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens > 0


async def test_mock_reply_echoes_last_user_message():
    provider = MockProvider(token_delay=0)
    result = await provider.complete(
        "mock-1",
        [ChatMessage("user", "first"), ChatMessage("user", "remember this")],
    )
    assert "remember this" in result.text
