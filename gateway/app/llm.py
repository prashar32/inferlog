"""LLM runtime: the model catalog and the inferlog client the gateway uses.

The catalog is intentionally a plain hard-coded list. Which models are
actually offered depends on which provider keys are configured — `mock`
is always available so the app works with no keys at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from inferlog import HttpSink, LoggedLLMClient, LogDispatcher
from inferlog.providers import MockProvider, Provider

from .config import Settings

log = logging.getLogger("gateway.llm")


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


class LLMRuntime:
    def __init__(
        self,
        client: LoggedLLMClient,
        dispatcher: LogDispatcher,
        providers: dict[str, Provider],
    ):
        self.client = client
        self._dispatcher = dispatcher
        self._providers = providers

    @classmethod
    def build(cls, settings: Settings) -> "LLMRuntime":
        providers: dict[str, Provider] = {"mock": MockProvider()}
        if settings.openai_api_key:
            from inferlog.providers import OpenAIProvider

            providers["openai"] = OpenAIProvider(api_key=settings.openai_api_key)
        if settings.anthropic_api_key:
            from inferlog.providers import AnthropicProvider

            providers["anthropic"] = AnthropicProvider(api_key=settings.anthropic_api_key)

        log.info("LLM providers enabled: %s", sorted(providers))
        sink = HttpSink(settings.ingest_url, api_key=settings.ingest_api_key)
        dispatcher = LogDispatcher(sink)
        client = LoggedLLMClient(
            service="chat-gateway", dispatcher=dispatcher, providers=providers
        )
        return cls(client, dispatcher, providers)

    def start(self) -> None:
        self._dispatcher.start()

    async def aclose(self) -> None:
        await self._dispatcher.aclose()

    @property
    def available_models(self) -> list[ModelOption]:
        return [m for m in CATALOG if m.provider in self._providers]

    def resolve(self, model_id: str) -> ModelOption | None:
        return next((m for m in self.available_models if m.model == model_id), None)

    def default_model(self) -> ModelOption:
        models = self.available_models
        # Prefer a real provider; fall back to mock when no keys are set.
        return next((m for m in models if m.provider != "mock"), models[-1])

    def logging_stats(self) -> dict:
        return self._dispatcher.stats()
