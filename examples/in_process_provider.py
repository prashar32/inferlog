"""In-process custom provider — what's in the customer's codebase
before and after adding inferlog.

Scenario: the customer's "model" is a Python object — embedded
inference (ONNX, llama.cpp, transformers), an in-house judge, or a
mock for tests. There is no HTTP traffic to hook into, so the
httpx-level capture won't see it. `inferlog.wrap_provider(...)` is
the supported path for this case.

The point of this file is to show, in one place, exactly what code
exists *before* and exactly what is added *after*. Nothing else.
"""

from __future__ import annotations

import os
from typing import AsyncIterator


# ════════════════════════════════════════════════════════════════════════
# BEFORE — the customer's in-process model, no inferlog.
# ════════════════════════════════════════════════════════════════════════
# A plain Python class with the customer's own interface. Could be
# anything — the only thing that matters is the customer already calls
# its `complete` / `stream` from their app.

from inferlog.providers import ChatMessage, Completion, Provider, StreamChunk, Usage


class MyEmbeddedModel(Provider):
    """The customer's own model. Subclass inferlog.providers.Provider so
    the SDK knows the call shape (`complete` / `stream` returning the
    standard Completion / StreamChunk types).
    """

    async def complete(self, model, messages, **opts) -> Completion:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        text = f"({model}) reply to: {last}"
        return Completion(
            text=text,
            usage=Usage(prompt_tokens=len(last.split()),
                        completion_tokens=len(text.split()),
                        total_tokens=len(last.split()) + len(text.split())),
        )

    async def stream(self, model, messages, **opts) -> AsyncIterator[StreamChunk]:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        for word in f"({model}) reply to: {last}".split():
            yield StreamChunk(text=word + " ")
        yield StreamChunk(usage=Usage(prompt_tokens=len(last.split()),
                                       completion_tokens=4,
                                       total_tokens=4 + len(last.split())))


# Before inferlog the customer was calling this directly:
#
#     model = MyEmbeddedModel()
#     reply = await model.complete("v1", [ChatMessage("user", "hi")])
#
# That works, but produces no telemetry.


# ════════════════════════════════════════════════════════════════════════
# AFTER — same Provider class, wrapped at startup. The customer's
# calling code becomes `client.complete(provider=..., model=..., ...)`
# instead of `model.complete(...)`; nothing else changes.
# ════════════════════════════════════════════════════════════════════════

import inferlog

inferlog.init(
    service="my-app",
    endpoint="https://ingest.ollive.ai/v1/ingest",
    api_key=os.environ["OLLIVE_API_KEY"],
    # Not strictly needed for this case; the in-process path doesn't
    # touch httpx at all. Off here for clarity.
    capture_all_httpx=False,
)

# One line — `wrap_provider` returns a LoggedLLMClient pre-wired to the
# global runtime's dispatcher + redactor. No reach into Runtime internals.
client = inferlog.wrap_provider(my_model=MyEmbeddedModel())


# Customer call site after the swap. The shape is `client.complete(
# provider=..., model=..., messages=...)` — same logical call, now logged.
async def run_example() -> None:
    reply = await client.complete(
        provider="my_model",
        model="v1",
        messages=[ChatMessage("user", "hi")],
    )
    print(reply.text)


# ════════════════════════════════════════════════════════════════════════
# The diff, in one sentence:
#   wrap once at startup, change the call site from `model.complete(...)`
#   to `client.complete(provider="my_model", ...)`. Everything else
#   (the Provider class, the messages format, the result type) is
#   identical to BEFORE.
#
# Each call produces an InferenceEvent with provider="my_model",
# the right token counts, PII-redacted previews, and any
# `inferlog.context(...)` tags active in scope.
# ════════════════════════════════════════════════════════════════════════
