# inferlog

A drop-in SDK that turns any LLM call into a structured inference log
without changing the host's code. One `inferlog.init(...)` at startup
and every model call going out through `httpx` is captured —
regardless of which library or which provider is in use.

The core stays small. Only required dependency: `httpx`. The SDK has no
runtime dependency on `openai`, `anthropic`, or any other vendor SDK —
capture is at the HTTP layer, so customers install whatever provider
library they actually use.

**Model-agnostic by construction.** OpenAI's SDK, Anthropic's SDK, raw
`httpx` calls to self-hosted models (vLLM, Ollama, llama.cpp),
OpenAI-compatible proxies (OpenRouter, Together, LiteLLM proxy),
LangChain, LlamaIndex — all the same code path. Provider handlers in
`parsers.py` recognise each shape by URL; add your own with
`inferlog.parsers.add_handler(...)` for a private model behind an
internal URL.

## Install

```bash
pip install inferlog
```

That's it — no provider extras. Install `openai` / `anthropic` /
whichever vendor SDKs your app actually uses directly.

## Use

### Mode 1 — global capture (default, recommended)

`init(capture_all_httpx=True)` (the default) patches `httpx` once at
startup. Every LLM call out of any library in the process is captured.

```python
import inferlog

inferlog.init(
    service="my-app",
    endpoint="https://ingest.example.com/v1/ingest",
    api_key="...",
)

# Existing code is unchanged. Native vendor SDKs, captured at the
# transport layer:
resp = await openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "..."}],
)
resp = await anthropic_client.messages.create(...)

# Optional: tag the scope so every inference inside it carries the tags.
with inferlog.context(conversation_id=cid, user_id=uid, tenant_id=tid):
    await openai_client.chat.completions.create(...)
```

Non-LLM httpx calls are not logged — the wrapper checks the URL against
the registered handlers and bails out immediately when nothing matches.

### Mode 2 — surgical capture (opt-in, per-client)

When you don't want our patch sitting on every httpx call in the
process — for example a service where most httpx traffic is non-LLM —
turn off the global patch and attach a transport to just the clients
you want captured.

```python
inferlog.init(..., capture_all_httpx=False)

openai_client = openai.AsyncOpenAI(
    api_key=...,
    http_client=httpx.AsyncClient(transport=inferlog.transport()),
)
# Other httpx clients in the process are untouched.
```

### Mode 3 — in-process custom providers

For models that don't speak HTTP (mocks, custom in-house inference),
`inferlog.wrap_provider(...)` plugs them into the same dispatcher and
redactor as the auto-captured path. No need to know about `Runtime`
internals.

```python
from inferlog.providers import ChatMessage, MockProvider

inferlog.init(...)
client = inferlog.wrap_provider(mock=MockProvider())

async for chunk in client.stream(provider="mock", model="mock-1",
                                 messages=[ChatMessage("user", "hello")]):
    print(chunk.text, end="")
```

A contextvar inside the wrapper prevents double-logging if the wrapped
provider ever does go over the wire.

## What gets captured

`InferenceEvent` fields:

| Field | Source |
|---|---|
| `request_id` | SDK (UUID); idempotency key on the ingestion side |
| `service`, `provider`, `model` | from `init()` and the call |
| `status` | `success` / `error` / `cancelled` |
| `streamed` | True / False |
| `started_at`, `completed_at`, `latency_ms` | timed by the SDK |
| `ttft_ms` | time to first token (streaming only) |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | from the provider's `usage` |
| `input_preview`, `output_preview` | **redacted in-process** |
| `error_type`, `error_message` | classified by the SDK into a stable set |
| `pii_redaction_count` | number of redaction substitutions |
| `tags` | from `inferlog.context(...)` |
| `client_metadata` | free-form, set by the SDK / caller |
| `sdk_version`, `schema_version` | version metadata |

## Production features

| Concern | What you get |
|---|---|
| **PII redaction** | In-process, before anything leaves the host. Default regex; `extra_patterns=...`; `custom=...` plug-in |
| **Sampling** | `KeepAll` (default), `Probability(rate)`, `AlwaysKeepErrors(inner)`, `CustomSampler(fn)` |
| **Backpressure visibility** | `on_drop(count, reason)` callback |
| **Network resilience** | Exponential backoff + jitter; honours `Retry-After` on 429 / 503 |
| **Auth schemes** | `auth_scheme="x-api-key"` (default) or `"bearer"` |
| **Per-event size caps** | `error_message`, `tags`, `client_metadata` bounded at `to_payload()` — a buggy call site can't ship multi-MB events that DoS ingestion |
| **Shutdown** | `atexit` flush; `await inferlog.ashutdown()` for clean async drain |
| **Self-observability** | `inferlog.stats()` → `{queued, delivered, dropped, closed}` |
| **Safety** | Every wrapper is in `try/except` — the SDK does not break the host call path |

### Examples

**Probability sampling, keep all errors:**
```python
from inferlog import AlwaysKeepErrors, Probability
inferlog.init(..., sampler=AlwaysKeepErrors(Probability(0.05)))
```

**See drops as they happen:**
```python
def on_drop(count, reason):
    metrics.increment("inferlog.dropped", count, tags={"reason": reason})

inferlog.init(..., on_drop=on_drop)
```

**Plug in a stronger redactor:**
```python
from inferlog import Redactor

def my_presidio_redactor(text: str) -> tuple[str, int]:
    result = analyzer.analyze(text=text, language="en")
    redacted = anonymizer.anonymize(text, result)
    return redacted.text, len(result)

inferlog.init(..., redactor=Redactor(custom=my_presidio_redactor))
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

66 unit tests covering:
* dispatcher (batching, retry, jitter, retry-after, drops, lazy start);
* the explicit-wrapper client and `wrap_provider` (cancellation, error
  classification, redaction, runtime wiring);
* sampling and the HTTP sink (auth schemes, transient errors);
* **HTTP-level capture across four provider shapes** — OpenAI,
  Anthropic, Ollama (NDJSON), and OpenAI-compatible — plus a raw httpx
  call with no SDK at all, and a negative test that non-LLM POSTs pass
  through uncaptured;
* the per-client transport mode (`inferlog.transport()`);
* per-event size caps (`error_message`, `tags`, `client_metadata`).
