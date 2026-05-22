"""Provider abstraction.

Every provider speaks the same small vocabulary so the rest of the SDK
(and the gateway) never branches on which vendor is in use. Adding a
provider means implementing `complete` and `stream` — nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


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
    """One streamed delta. `usage` is usually only set on the final chunk."""

    text: str = ""
    usage: Usage | None = None


@dataclass
class Completion:
    """Result of a non-streaming call."""

    text: str
    usage: Usage = field(default_factory=Usage)


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(
        self, model: str, messages: list[ChatMessage], **opts
    ) -> Completion: ...

    @abstractmethod
    def stream(
        self, model: str, messages: list[ChatMessage], **opts
    ) -> AsyncIterator[StreamChunk]:
        """Return an async iterator of StreamChunks (implemented as an
        async generator in concrete providers)."""
        ...
