"""Custom HTTP provider — what's in the customer's codebase before and
after adding inferlog.

Scenario: the customer runs their own LLM at https://llm.acme.corp/v2/generate
with a wire format that isn't OpenAI / Anthropic / Ollama. Imaginary shapes:

    request  = {"prompt": str, "model": str, "stream": bool}
    response = {"completion": str, "input_tokens": int, "output_tokens": int}

The point of this file is to show, in one place, exactly what code
exists *before* and exactly what is added *after*. Nothing else.
"""

from __future__ import annotations

import os
import httpx


# ════════════════════════════════════════════════════════════════════════
# BEFORE — what's in the customer's codebase today, no inferlog.
# ════════════════════════════════════════════════════════════════════════
# A plain httpx call. The customer might wrap it in a function, a class,
# a LangChain BaseLLM — shape doesn't matter. The bytes on the wire are
# what inferlog will hook into later.

async def call_acme_model(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://llm.acme.corp/v2/generate",
            json={"model": "acme-7b", "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["completion"]


# ════════════════════════════════════════════════════════════════════════
# AFTER — same codebase + inferlog. The customer's call site (above)
# is UNCHANGED. The only additions are (A) a handler class and (B) two
# lines at startup.
# ════════════════════════════════════════════════════════════════════════

# ---- (A) one handler class, written ONCE, in any file -----------------
# Tells inferlog how to recognise this URL and how to parse Acme's
# request / response shapes into the standard InferenceEvent fields.

import inferlog
from inferlog.parsers import (
    ProviderHandler, RequestMeta, ResponseMeta, StreamParser,
    _safe_json, add_handler,
)


class AcmeInternalHandler(ProviderHandler):
    name = "acme-internal"          # this is what appears as `provider` on the event

    def matches(self, request: httpx.Request) -> bool:
        # Keep this specific so it doesn't accidentally catch unrelated POSTs.
        return (
            request.method == "POST"
            and request.url.host == "llm.acme.corp"
            and request.url.path.endswith("/v2/generate")
        )

    def parse_request(self, request: httpx.Request) -> RequestMeta:
        body = _safe_json(request.content) or {}
        return RequestMeta(
            model=str(body.get("model", "unknown")),
            streaming=bool(body.get("stream")),
            input_text=body.get("prompt"),     # becomes input_preview (PII-redacted)
        )

    def parse_response(self, response: httpx.Response) -> ResponseMeta:
        body = _safe_json(response.content) or {}
        prompt = body.get("input_tokens")
        completion = body.get("output_tokens")
        return ResponseMeta(
            output_text=body.get("completion"),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=(
                prompt + completion
                if prompt is not None and completion is not None
                else None
            ),
        )

    def make_stream_parser(self) -> StreamParser:
        # Only called when parse_request set streaming=True. If your model
        # never streams, raising here is fine — it will never be invoked.
        # For a streaming model, subclass StreamParser and accumulate
        # `output_text` / token counts as chunks arrive. See
        # `_OllamaStreamParser` in sdk/inferlog/parsers.py for a short
        # NDJSON reference.
        raise NotImplementedError("Acme model is non-streaming in this example")


# ---- (B) two lines at process startup, before any LLM call ------------

add_handler(AcmeInternalHandler())
inferlog.init(
    service="my-app",
    endpoint="https://ingest.ollive.ai/v1/ingest",
    api_key=os.environ["OLLIVE_API_KEY"],
)


# ════════════════════════════════════════════════════════════════════════
# The diff, in one sentence:
#   one handler class + two startup lines.
#   `call_acme_model(...)` is byte-identical to the BEFORE version.
#
# Every call from anywhere in the codebase to llm.acme.corp/v2/generate
# now produces an InferenceEvent with provider="acme-internal",
# model="acme-7b", parsed token counts, PII-redacted previews, and any
# tags from `inferlog.context(...)` scopes the call happens to be in.
# ════════════════════════════════════════════════════════════════════════
