"""Provider primitives used by the explicit-wrapper path (LoggedLLMClient).

The OpenAI and Anthropic adapters that used to live here were removed in
v0.3 — model-agnostic capture is now HTTP-level (see `inferlog.auto`), so
the customer uses the native vendor SDKs directly and we capture below
them. The only adapter kept here is `MockProvider`, used by the demo's
offline path (which doesn't go over HTTP).

To register a fully custom in-process provider, subclass `Provider` and
pass it to `LoggedLLMClient(providers={"my-llm": MyProvider()})`.
"""

from .base import ChatMessage, Completion, Provider, StreamChunk, Usage
from .mock import MockProvider

__all__ = [
    "ChatMessage",
    "Completion",
    "Provider",
    "StreamChunk",
    "Usage",
    "MockProvider",
]

