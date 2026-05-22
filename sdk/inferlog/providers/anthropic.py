"""Anthropic adapter.

Mostly here to prove the provider abstraction holds for more than one
vendor. Anthropic differs from OpenAI in two ways the adapter smooths over:
the system prompt is a separate argument, and `max_tokens` is required.
"""

from __future__ import annotations

from typing import AsyncIterator

from .base import ChatMessage, Completion, Provider, StreamChunk, Usage

_DEFAULT_MAX_TOKENS = 1024


def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
    system = next((m.content for m in messages if m.role == "system"), None)
    turns = [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role in ("user", "assistant")
    ]
    return system, turns


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "AnthropicProvider needs the anthropic package: "
                "pip install inferlog[anthropic]"
            ) from exc
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self, model: str, messages: list[ChatMessage], **opts
    ) -> Completion:
        system, turns = _split_system(messages)
        opts.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
        resp = await self._client.messages.create(
            model=model, system=system or "", messages=turns, **opts
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = Usage(
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            resp.usage.input_tokens + resp.usage.output_tokens,
        )
        return Completion(text=text, usage=usage)

    async def stream(
        self, model: str, messages: list[ChatMessage], **opts
    ) -> AsyncIterator[StreamChunk]:
        system, turns = _split_system(messages)
        opts.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
        prompt_tokens = 0
        async with self._client.messages.stream(
            model=model, system=system or "", messages=turns, **opts
        ) as stream:
            async for event in stream:
                if event.type == "message_start":
                    prompt_tokens = event.message.usage.input_tokens
                elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield StreamChunk(text=event.delta.text)
                elif event.type == "message_delta":
                    completion = event.usage.output_tokens
                    yield StreamChunk(
                        usage=Usage(prompt_tokens, completion, prompt_tokens + completion)
                    )
