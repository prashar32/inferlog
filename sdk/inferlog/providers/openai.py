"""OpenAI adapter.

Imports the `openai` package lazily so the SDK core has no hard dependency
on it — install with `pip install inferlog[openai]`.
"""

from __future__ import annotations

from typing import AsyncIterator

from .base import ChatMessage, Completion, Provider, StreamChunk, Usage


def _to_openai(messages: list[ChatMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, api_key: str, base_url: str | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "OpenAIProvider needs the openai package: pip install inferlog[openai]"
            ) from exc
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self, model: str, messages: list[ChatMessage], **opts
    ) -> Completion:
        resp = await self._client.chat.completions.create(
            model=model, messages=_to_openai(messages), **opts
        )
        choice = resp.choices[0]
        usage = Usage()
        if resp.usage:
            usage = Usage(
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
                resp.usage.total_tokens,
            )
        return Completion(text=choice.message.content or "", usage=usage)

    async def stream(
        self, model: str, messages: list[ChatMessage], **opts
    ) -> AsyncIterator[StreamChunk]:
        # include_usage makes OpenAI emit a final chunk with token counts;
        # without it, streamed responses have no usage data at all.
        stream = await self._client.chat.completions.create(
            model=model,
            messages=_to_openai(messages),
            stream=True,
            stream_options={"include_usage": True},
            **opts,
        )
        async for chunk in stream:
            usage = None
            if chunk.usage:
                usage = Usage(
                    chunk.usage.prompt_tokens,
                    chunk.usage.completion_tokens,
                    chunk.usage.total_tokens,
                )
            delta = ""
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
            if delta or usage:
                yield StreamChunk(text=delta, usage=usage)
