# inferlog

A drop-in SDK that turns LLM provider calls into structured inference
logs. The host app does not have to change its code — one call to
`inferlog.init(...)` patches the OpenAI and Anthropic clients and every
subsequent call is captured.

The core stays small. Only required dependency: `httpx`.

## Install

```bash
pip install inferlog                  # core only
pip install "inferlog[openai]"        # + the OpenAI Python SDK
pip install "inferlog[anthropic]"     # + the Anthropic Python SDK
```

## Use

### Auto-instrumentation (recommended)

```python
import inferlog

inferlog.init(
    service="my-app",
    endpoint="https://ingest.example.com/v1/ingest",
    api_key="...",
)

# Your existing code is unchanged:
resp = await openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "..."}],
)

# Optional: tag the scope so every inference inside it carries the tags.
with inferlog.context(conversation_id=cid, user_id=uid, tenant_id=tid):
    await openai_client.chat.completions.create(...)
    await anthropic_client.messages.create(...)
```

### Explicit wrapper (for custom or in-house providers)

```python
from inferlog import LoggedLLMClient, get_runtime
from inferlog.providers import ChatMessage, MockProvider

# init() set up the global dispatcher + redactor; reuse them here.
rt = get_runtime()
client = LoggedLLMClient(
    service="my-app",
    dispatcher=rt.dispatcher,
    redactor=rt.redactor,
    providers={"mock": MockProvider()},
)

async for chunk in client.stream(provider="mock", model="mock-1",
                                 messages=[ChatMessage("user", "hello")]):
    print(chunk.text, end="")
```

The two paths produce identically-shaped events. A contextvar inside the
explicit wrapper prevents double-logging when both paths are active.

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

42 unit tests covering the dispatcher (batching, retry, jitter,
retry-after, drops, lazy start), the client wrapper (cancellation,
error classification, redaction), sampling, the HTTP sink (auth schemes,
transient errors), and auto-instrumentation (success / error / streaming
/ context tags) against a mocked OpenAI HTTP layer.
