"""LLM runtime: the model catalog and the chat-provider abstraction.

In v0.3 the gateway demonstrates the SDK's HTTP-level capture: for OpenAI
and Anthropic we instantiate the **native** SDK clients (no inferlog
wrappers). inferlog's `httpx` patches intercept those calls and emit
events automatically. The customer's experience is identical to ours.

For the offline `mock` provider — which doesn't go over HTTP and so
isn't reachable by HTTP-level capture — we use inferlog's explicit
`LoggedLLMClient` path. A contextvar inside that wrapper suppresses HTTP
capture so we don't double-log if it ever did go over the wire.
"""

from __future__ import annotations

import logging
import httpx
from dataclasses import dataclass
from typing import AsyncIterator

import inferlog
from inferlog import LoggedLLMClient
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

        self._client = openai.AsyncOpenAI(api_key=api_key, http_client=httpx.AsyncClient(transport=inferlog.transport()))

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

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

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
    capture won't see it. The explicit-wrapper path emits events for us."""

    def __init__(self):
        rt = inferlog.get_runtime()
        assert rt is not None, "inferlog.init() must be called before MockChatProvider"
        self._client = LoggedLLMClient(
            service=rt.service,
            dispatcher=rt.dispatcher,
            redactor=rt.redactor,
            providers={"mock": MockProvider(token_delay=0.02)},
        )

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
        # Single line — patches httpx, sets up dispatcher + redactor +
        # sampler. From here on, every LLM call going through httpx is
        # captured automatically, no matter which library makes it.
        installed = inferlog.init(
            service="chat-gateway",
            endpoint=settings.ingest_url,
            api_key=settings.ingest_api_key,
            capture_all_httpx=False,
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
