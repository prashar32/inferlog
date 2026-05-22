from .base import ChatMessage, Completion, Provider, StreamChunk, Usage
from .mock import MockProvider

# OpenAI / Anthropic adapters are imported lazily: importing this package
# should not require the vendor SDKs to be installed.

__all__ = [
    "ChatMessage",
    "Completion",
    "Provider",
    "StreamChunk",
    "Usage",
    "MockProvider",
    "OpenAIProvider",
    "AnthropicProvider",
]


def __getattr__(name: str):
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
