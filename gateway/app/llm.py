"""LLM runtime: the model catalog and the inferlog client the gateway uses.

The catalog is intentionally a plain hard-coded list. Which models are
actually offered depends on which provider keys are configured — `mock`
is always available so the app works with no keys at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import inferlog
from inferlog import LoggedLLMClient
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
        providers: dict[str, Provider],
    ):
        self.client = client
        self._providers = providers

    @classmethod
    def build(cls, settings: Settings) -> "LLMRuntime":
        # One-line SDK init — sets up the global dispatcher + redactor, AND
        # auto-instruments any openai/anthropic clients used elsewhere in the
        # process. PII redaction happens here, before anything leaves us.
        installed = inferlog.init(
            service="chat-gateway",
            endpoint=settings.ingest_url,
            api_key=settings.ingest_api_key,
        )
        log.info("inferlog initialised; auto-instrumented providers: %s", installed)

        rt = inferlog.get_runtime()
        assert rt is not None  # init() just set it

        providers: dict[str, Provider] = {"mock": MockProvider()}
        if settings.openai_api_key:
            from inferlog.providers import OpenAIProvider

            providers["openai"] = OpenAIProvider(api_key=settings.openai_api_key)
        if settings.anthropic_api_key:
            from inferlog.providers import AnthropicProvider

            providers["anthropic"] = AnthropicProvider(api_key=settings.anthropic_api_key)

        log.info("chat providers enabled: %s", sorted(providers))

        # The explicit LoggedLLMClient shares the global dispatcher and
        # redactor — both integration paths produce identically-shaped
        # events. Inside this client, auto-instrumentation is suppressed
        # via a contextvar to avoid double-logging.
        client = LoggedLLMClient(
            service="chat-gateway",
            dispatcher=rt.dispatcher,
            providers=providers,
            redactor=rt.redactor,
        )
        return cls(client, providers)

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
        # Prefer a real provider; fall back to mock when no keys are set.
        return next((m for m in models if m.provider != "mock"), models[-1])

    def logging_stats(self) -> dict:
        return inferlog.stats()
