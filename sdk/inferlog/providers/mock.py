"""A deterministic offline provider.

This is what lets the whole system — chat, streaming, ingestion,
dashboards — run and be tested without any API key. It also keeps the
test suite fast and free.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .base import ChatMessage, Completion, Provider, StreamChunk, Usage


def _word_count(text: str) -> int:
    return max(len(text.split()), 1)


class MockProvider(Provider):
    name = "mock"

    def __init__(self, reply: str | None = None, token_delay: float = 0.03):
        # token_delay simulates network/generation latency so the streaming
        # UI and the latency dashboard show something realistic.
        self._reply = reply
        self._token_delay = token_delay

    def _reply_for(self, messages: list[ChatMessage]) -> str:
        if self._reply is not None:
            return self._reply
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        return (
            f"You said: \"{last_user.strip()[:160]}\". "
            "I'm the offline mock model, so this reply is canned — but the "
            "logging pipeline around it is the real thing."
        )

    def _usage(self, messages: list[ChatMessage], reply: str) -> Usage:
        prompt = sum(_word_count(m.content) for m in messages)
        completion = _word_count(reply)
        return Usage(prompt, completion, prompt + completion)

    async def complete(
        self, model: str, messages: list[ChatMessage], **opts
    ) -> Completion:
        reply = self._reply_for(messages)
        await asyncio.sleep(self._token_delay * 4)
        return Completion(text=reply, usage=self._usage(messages, reply))

    async def stream(
        self, model: str, messages: list[ChatMessage], **opts
    ) -> AsyncIterator[StreamChunk]:
        reply = self._reply_for(messages)
        words = reply.split()
        for i, word in enumerate(words):
            await asyncio.sleep(self._token_delay)
            suffix = " " if i < len(words) - 1 else ""
            yield StreamChunk(text=word + suffix)
        # Final chunk carries usage, mirroring how real providers behave.
        yield StreamChunk(usage=self._usage(messages, reply))
