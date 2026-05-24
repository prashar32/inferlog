"""LLM runtime: the model catalog and the chat-provider abstraction.

The gateway intentionally demonstrates ALL THREE inferlog integration
modes side-by-side so it doubles as a customer-facing reference:

  * OpenAI provider  → Mode 1 (global capture). Native SDK client, no
    inferlog plumbing at the call site. Captured because
    `inferlog.init(capture_all_httpx=True)` patched httpx at startup.

  * Anthropic provider → Mode 2 (per-client transport). Native SDK
    client constructed with `http_client=httpx.AsyncClient(
    transport=inferlog.transport())`. The transport is the surgical
    opt-in: it makes capture explicit at the call site. When the global
    patch is also active (the default), the transport defers to it to
    avoid double-logging — flip `capture_all_httpx=False` at init and
    only this client is captured.

  * Mock provider → Mode 3 (in-process wrapper). The mock doesn't go
    over HTTP at all, so neither of the httpx-based modes can see it.
    `inferlog.wrap_provider(mock=MockProvider())` plugs it into the
    same dispatcher + redactor; a contextvar prevents double-logging.

All three paths produce identically-shaped `InferenceEvent`s.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator

import httpx
import inferlog
from inferlog.providers import ChatMessage as SDKChatMessage, MockProvider

from .config import Settings

log = logging.getLogger("gateway.llm")


# --- catalog ----------------------------------------------------------


@dataclass(frozen=True)
class ModelOption:
    provider: str
    model: str
    label: str


# Order matters: the first available non-mock entry becomes the UI default.
# gpt-4o-mini is first because it's the most broadly accessible OpenAI model
# — newer keys / projects often aren't entitled to the gpt-4.1 family yet.
CATALOG: list[ModelOption] = [
    ModelOption("openai", "gpt-4o-mini", "GPT-4o mini"),
    ModelOption("openai", "gpt-4.1-mini", "GPT-4.1 mini"),
    ModelOption("openai", "gpt-4.1", "GPT-4.1"),
    ModelOption("anthropic", "claude-sonnet-4-5", "Claude Sonnet 4.5"),
    ModelOption("mock", "mock-1", "Mock model (offline)"),
]


# --- gateway-internal chat interface ---------------------------------
# This is the shape `chat.py` iterates over. Each provider yields these,
# regardless of whether under the hood it's OpenAI / Anthropic / mock.


@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class StreamChunk:
    text: str = ""
    usage: Usage | None = None


class ChatProvider:
    """A normalised streaming chat interface. Concrete implementations
    call their respective native client (or the explicit-wrapper mock)
    and yield `StreamChunk`s."""

    async def stream(
        self, model: str, messages: list[ChatMessage], **opts
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError
        yield  # type: ignore[unreachable]  # keep it an async generator signature


# --- OpenAI (HTTP-captured) ------------------------------------------


class OpenAIChatProvider(ChatProvider):
    def __init__(self, api_key: str):
        import openai

        # Native client, no plumbing — inferlog's global httpx patch
        # captures this call at the transport layer.
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def stream(self, model, messages, **opts):
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            stream_options={"include_usage": True},
            **opts,
        )
        async for chunk in stream:
            text = ""
            usage: Usage | None = None
            if chunk.choices:
                text = chunk.choices[0].delta.content or ""
            if chunk.usage:
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
            if text or usage:
                yield StreamChunk(text=text, usage=usage)


# --- Anthropic (HTTP-captured) ---------------------------------------


class AnthropicChatProvider(ChatProvider):
    def __init__(self, api_key: str):
        import anthropic

        # Mode 2 (per-client transport) demo. The customer constructs a
        # native Anthropic client but passes an httpx client whose
        # transport is `inferlog.transport()`. This is the surgical
        # opt-in — capture is wired here, visibly, instead of through a
        # process-wide patch. With `capture_all_httpx=True` (the default
        # in this demo) the global patch is active and this transport
        # gracefully steps aside; set `capture_all_httpx=False` at init
        # to make this transport the *only* capture path for this client.
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            http_client=httpx.AsyncClient(transport=inferlog.transport()),
        )

    async def stream(self, model, messages, max_tokens: int = 1024, **opts):
        system = next((m.content for m in messages if m.role == "system"), None)
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        prompt_tokens = 0
        async with self._client.messages.stream(
            model=model,
            system=system or "",
            messages=turns,
            max_tokens=max_tokens,
            **opts,
        ) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "message_start":
                    prompt_tokens = event.message.usage.input_tokens
                elif etype == "content_block_delta" and event.delta.type == "text_delta":
                    yield StreamChunk(text=event.delta.text)
                elif etype == "message_delta":
                    completion = event.usage.output_tokens
                    yield StreamChunk(
                        usage=Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion,
                            total_tokens=prompt_tokens + completion,
                        )
                    )


# --- Mock (explicit-wrapper path) ------------------------------------


class MockChatProvider(ChatProvider):
    """The mock model is in-process; nothing leaves the host, so HTTP
    capture won't see it. `inferlog.wrap_provider()` plugs it into the
    same dispatcher + redactor the auto-instrumentation path uses."""

    def __init__(self):
        self._client = inferlog.wrap_provider(mock=MockProvider(token_delay=0.02))

    async def stream(self, model, messages, **opts):
        sdk_messages = [SDKChatMessage(m.role, m.content) for m in messages]
        async for chunk in self._client.stream(
            provider="mock",
            model=model,
            messages=sdk_messages,
            metadata={"channel": "web"},
        ):
            usage: Usage | None = None
            if chunk.usage is not None:
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
            yield StreamChunk(text=chunk.text or "", usage=usage)


# --- runtime ----------------------------------------------------------


class LLMRuntime:
    def __init__(self, providers: dict[str, ChatProvider]):
        self._providers = providers

    @classmethod
    def build(cls, settings: Settings) -> "LLMRuntime":
        # ──────────────────────────────────────────────────────────────
        # inferlog SDK — single-line init.
        #
        # `capture_all_httpx=True` (the default; spelt out here for
        # clarity) globally patches httpx.AsyncClient.send and
        # httpx.Client.send. After this returns, every LLM HTTP call the
        # process makes — via any library — is captured automatically.
        # Non-LLM httpx traffic is recognised by URL and passes through
        # untouched (see sdk/inferlog/parsers.py — handlers check host /
        # path and return None for anything they don't recognise).
        #
        # This is the recommended customer integration. The alternative
        # (per-client `inferlog.transport()`) is also supported and is
        # demonstrated by AnthropicChatProvider above — useful when a
        # customer cannot accept a process-wide monkey-patch on httpx
        # and prefers to mark each captured client explicitly.
        # ──────────────────────────────────────────────────────────────
        installed = inferlog.init(
            service="chat-gateway",
            endpoint=settings.ingest_url,
            api_key=settings.ingest_api_key,
            capture_all_httpx=True,
        )
        log.info("inferlog initialised; transports patched: %s", installed)

        providers: dict[str, ChatProvider] = {"mock": MockChatProvider()}
        if settings.openai_api_key:
            providers["openai"] = OpenAIChatProvider(settings.openai_api_key)
        if settings.anthropic_api_key:
            providers["anthropic"] = AnthropicChatProvider(settings.anthropic_api_key)
        log.info("chat providers enabled: %s", sorted(providers))
        return cls(providers)

    def start(self) -> None:
        # init() already started the dispatcher; kept for API compatibility.
        pass

    async def aclose(self) -> None:
        await inferlog.ashutdown()

    @property
    def available_models(self) -> list[ModelOption]:
        return [m for m in CATALOG if m.provider in self._providers]

    def resolve(self, model_id: str) -> ModelOption | None:
        return next((m for m in self.available_models if m.model == model_id), None)

    def default_model(self) -> ModelOption:
        models = self.available_models
        return next((m for m in models if m.provider != "mock"), models[-1])

    def logging_stats(self) -> dict:
        return inferlog.stats()

    def stream(
        self, provider: str, model: str, messages: list[ChatMessage], **opts
    ) -> AsyncIterator[StreamChunk]:
        prov = self._providers.get(provider)
        if prov is None:
            raise ValueError(
                f"unknown provider {provider!r}; available: {sorted(self._providers)}"
            )
        return prov.stream(model, messages, **opts)
