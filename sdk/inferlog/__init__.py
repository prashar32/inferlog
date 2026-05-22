"""inferlog — a thin wrapper that turns any LLM call into a structured
inference log without the caller having to think about it.

Typical use (see the gateway service for a real integration):

    from inferlog import LoggedLLMClient, LogDispatcher, HttpSink
    from inferlog.providers import OpenAIProvider, MockProvider

    dispatcher = LogDispatcher(HttpSink("http://ingestion:8080/v1/ingest"))
    dispatcher.start()

    client = LoggedLLMClient(
        service="chat-gateway",
        dispatcher=dispatcher,
        providers={"openai": OpenAIProvider(api_key=...), "mock": MockProvider()},
    )

    async for chunk in client.stream(provider="openai", model="gpt-4.1-mini",
                                     messages=[...], conversation_id=cid):
        ...
"""

from .client import LoggedLLMClient
from .dispatcher import HttpSink, LogDispatcher, MemorySink, NullSink
from .events import SDK_VERSION, InferenceEvent
from .providers import ChatMessage, Completion, StreamChunk, Usage

__version__ = SDK_VERSION

__all__ = [
    "LoggedLLMClient",
    "LogDispatcher",
    "HttpSink",
    "MemorySink",
    "NullSink",
    "InferenceEvent",
    "SDK_VERSION",
    "ChatMessage",
    "Completion",
    "StreamChunk",
    "Usage",
]
