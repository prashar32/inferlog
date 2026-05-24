"""Per-provider parsers for HTTP-level instrumentation.

Each handler knows how to recognise its own provider's HTTP requests and
extract the metadata we care about: model name, the last user message,
streaming flag, output text, and token counts.

Adding a new provider = one handler class + one stream parser, registered
in `HANDLERS`. The customer can also call `add_handler(...)` to register
their own at runtime (e.g., for an internal LLM behind a private URL).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx


# ---------------------------------------------------------------- types


@dataclass
class RequestMeta:
    model: str
    streaming: bool
    input_text: str | None = None


@dataclass
class ResponseMeta:
    output_text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class StreamParser:
    """Accumulates state from a streaming response body. Subclasses
    override `feed()` to handle SSE / NDJSON / whatever the provider
    speaks.

    `completed` is True once a natural end-of-stream marker is observed
    (e.g. OpenAI's `[DONE]`, Anthropic's `message_stop`, Ollama's
    `done:true`). The wrapper uses it to distinguish "consumer cancelled
    mid-stream" from "consumer broke after the stream completed".
    """

    def __init__(self) -> None:
        self.output_text: str = ""
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.total_tokens: int | None = None
        self.completed: bool = False
        self._buffer: bytes = b""

    def feed(self, chunk: bytes) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------- helpers


def _safe_json(content: bytes | str | None) -> dict | None:
    if not content:
        return None
    try:
        if isinstance(content, str):
            return json.loads(content)
        return json.loads(content)
    except Exception:  # noqa: BLE001
        return None


def _last_user_text(messages: Any) -> str | None:
    """Defensive extraction of the last user message — never raise."""
    if not messages:
        return None
    try:
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # multimodal content list
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") in (None, "text"):
                        parts.append(str(p.get("text", "")))
                return " ".join(parts).strip() or None
    except Exception:  # noqa: BLE001
        return None
    return None


# ---------------------------------------------------------------- handler base


class ProviderHandler(ABC):
    name: str

    @abstractmethod
    def matches(self, request: httpx.Request) -> bool: ...

    @abstractmethod
    def parse_request(self, request: httpx.Request) -> RequestMeta: ...

    @abstractmethod
    def parse_response(self, response: httpx.Response) -> ResponseMeta: ...

    @abstractmethod
    def make_stream_parser(self) -> StreamParser: ...


# ---------------------------------------------------------------- OpenAI


class OpenAIHandler(ProviderHandler):
    name = "openai"

    def matches(self, request: httpx.Request) -> bool:
        if request.method != "POST":
            return False
        host = request.url.host or ""
        path = request.url.path or ""
        return "openai.com" in host and path.endswith("/chat/completions")

    def parse_request(self, request: httpx.Request) -> RequestMeta:
        body = _safe_json(request.content) or {}
        return RequestMeta(
            model=str(body.get("model", "unknown")),
            streaming=bool(body.get("stream")),
            input_text=_last_user_text(body.get("messages")),
        )

    def parse_response(self, response: httpx.Response) -> ResponseMeta:
        body = _safe_json(response.content) or {}
        choices = body.get("choices") or []
        text = None
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content")
        usage = body.get("usage") or {}
        return ResponseMeta(
            output_text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    def make_stream_parser(self) -> StreamParser:
        return _OpenAIStreamParser()


class _OpenAIStreamParser(StreamParser):
    """OpenAI streaming is SSE: lines like `data: {…}` separated by blank lines."""

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                self.completed = True
                continue
            obj = _safe_json(data)
            if obj is None:
                continue
            for choice in obj.get("choices", []) or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    self.output_text += content
            usage = obj.get("usage")
            if usage:
                self.prompt_tokens = usage.get("prompt_tokens")
                self.completion_tokens = usage.get("completion_tokens")
                self.total_tokens = usage.get("total_tokens")


# ---------------------------------------------------------------- Anthropic


class AnthropicHandler(ProviderHandler):
    name = "anthropic"

    def matches(self, request: httpx.Request) -> bool:
        if request.method != "POST":
            return False
        host = request.url.host or ""
        path = request.url.path or ""
        return "anthropic" in host and path.endswith("/v1/messages")

    def parse_request(self, request: httpx.Request) -> RequestMeta:
        body = _safe_json(request.content) or {}
        return RequestMeta(
            model=str(body.get("model", "unknown")),
            streaming=bool(body.get("stream")),
            input_text=_last_user_text(body.get("messages")),
        )

    def parse_response(self, response: httpx.Response) -> ResponseMeta:
        body = _safe_json(response.content) or {}
        text_parts = []
        for block in body.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        text = "".join(text_parts) or None
        usage = body.get("usage") or {}
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        total = (prompt + completion) if (prompt is not None and completion is not None) else None
        return ResponseMeta(
            output_text=text,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    def make_stream_parser(self) -> StreamParser:
        return _AnthropicStreamParser()


class _AnthropicStreamParser(StreamParser):
    """Anthropic SSE — events with `type` field, usage split across
    `message_start` (input_tokens) and `message_delta` (output_tokens)."""

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            obj = _safe_json(data)
            if obj is None:
                continue
            etype = obj.get("type")
            if etype == "message_start":
                msg = obj.get("message") or {}
                usage = msg.get("usage") or {}
                if usage.get("input_tokens") is not None:
                    self.prompt_tokens = usage["input_tokens"]
            elif etype == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta":
                    self.output_text += str(delta.get("text", ""))
            elif etype == "message_delta":
                usage = obj.get("usage") or {}
                if usage.get("output_tokens") is not None:
                    self.completion_tokens = usage["output_tokens"]
                    if self.prompt_tokens is not None:
                        self.total_tokens = self.prompt_tokens + self.completion_tokens
            elif etype == "message_stop":
                self.completed = True


# ---------------------------------------------------------------- Ollama (self-hosted)


class OllamaHandler(ProviderHandler):
    """Catches Ollama / OpenAI-incompatible self-hosted models that use
    `POST /api/chat` or `/api/generate` with NDJSON streaming."""

    name = "ollama"

    def matches(self, request: httpx.Request) -> bool:
        if request.method != "POST":
            return False
        path = request.url.path or ""
        return path.endswith("/api/chat") or path.endswith("/api/generate")

    def parse_request(self, request: httpx.Request) -> RequestMeta:
        body = _safe_json(request.content) or {}
        input_text = _last_user_text(body.get("messages"))
        if input_text is None and isinstance(body.get("prompt"), str):
            input_text = body["prompt"]
        return RequestMeta(
            model=str(body.get("model", "unknown")),
            # Ollama streams by default — only `stream=false` opts out.
            streaming=body.get("stream", True) is not False,
            input_text=input_text,
        )

    def parse_response(self, response: httpx.Response) -> ResponseMeta:
        body = _safe_json(response.content) or {}
        msg = body.get("message") or {}
        text = msg.get("content") or body.get("response")
        prompt = body.get("prompt_eval_count")
        completion = body.get("eval_count")
        total = (prompt + completion) if (prompt is not None and completion is not None) else None
        return ResponseMeta(
            output_text=text,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    def make_stream_parser(self) -> StreamParser:
        return _OllamaStreamParser()


class _OllamaStreamParser(StreamParser):
    """NDJSON — newline-delimited JSON, not SSE."""

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            obj = _safe_json(line)
            if obj is None:
                continue
            msg = obj.get("message") or {}
            content = msg.get("content") or obj.get("response", "")
            if content:
                self.output_text += str(content)
            if obj.get("done"):
                self.completed = True
                if obj.get("prompt_eval_count") is not None:
                    self.prompt_tokens = obj["prompt_eval_count"]
                if obj.get("eval_count") is not None:
                    self.completion_tokens = obj["eval_count"]
                if self.prompt_tokens is not None and self.completion_tokens is not None:
                    self.total_tokens = self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------- OpenAI-compatible
# vLLM, OpenRouter, Together, Fireworks, LiteLLM proxy, Anyscale, …
# These speak the OpenAI Chat Completions wire format on arbitrary hosts.


class OpenAICompatibleHandler(OpenAIHandler):
    """OpenAI wire format on a non-openai.com host. Same parser; the
    `provider` field in the event will say `openai_compatible` so you
    can still slice by host on the dashboard."""

    name = "openai_compatible"

    def matches(self, request: httpx.Request) -> bool:
        if request.method != "POST":
            return False
        host = request.url.host or ""
        path = request.url.path or ""
        # Skip the dedicated OpenAI / Anthropic handlers — they ran first.
        if "openai.com" in host or "anthropic" in host:
            return False
        return path.endswith("/v1/chat/completions") or path.endswith("/chat/completions")


# ---------------------------------------------------------------- registry

# Order matters — first match wins. Specific hosts before the generic
# OpenAI-compatible fallback.
HANDLERS: list[ProviderHandler] = [
    OpenAIHandler(),
    AnthropicHandler(),
    OllamaHandler(),
    OpenAICompatibleHandler(),
]


def add_handler(handler: ProviderHandler, *, position: int = 0) -> None:
    """Public extension point — customers register their own provider
    handler at init time (e.g., for an internal LLM behind a private URL,
    or a Bedrock-specific parser)."""
    HANDLERS.insert(position, handler)
