# InferLog

A small chatbot with a real inference-logging pipeline behind it, and a
drop-in SDK that captures every LLM call without the caller having to
change their code.

You chat with an LLM in the browser. Inside the chat backend a one-line
`inferlog.init(...)` patches `httpx`, so every outgoing LLM HTTP call
— OpenAI SDK, Anthropic SDK, raw httpx to Ollama / vLLM / OpenAI-compatible
proxies — is captured at the transport layer. Each event records
latency, tokens, status, previews, with PII redacted **before** the
event leaves the process. The event ships to an ingestion service that
validates, enriches with cost / throughput, and stores it. A dashboard
reads that back as latency, throughput and error charts.

The whole thing runs with one command and no API keys (there's a
built-in mock model); add an `OPENAI_API_KEY` to talk to a real model.

---

## Quick start

```bash
cp .env.example .env        # optional — add OPENAI_API_KEY here for GPT models
docker compose up --build   # or: make up
```

Then open **http://localhost:8088**.

- **Chat tab** — start a conversation, watch the answer stream in, hit
  **Stop** to cancel mid-stream. The left rail lists conversations; click
  one to resume it.
- **Dashboard tab** — latency / throughput / error charts over the logs
  the SDK produced.

Want the dashboard populated immediately? `make seed` pushes ~80 synthetic
inference events through the real pipeline.

To run the tests: `make test`. To tear down: `make down` (`make clean` also
drops the database volume).

> Without a provider key the UI offers a single **Mock model (offline)** —
> a deterministic local model. Everything else (streaming, logging,
> ingestion, dashboards, cancellation) is fully real on the mock model.

---

## What's in the box

| Requirement | Where |
|---|---|
| Multi-turn chatbot with a UI | `gateway/` + `web/` |
| Lightweight logging SDK / wrapper | `sdk/` (`inferlog`) |
| Ingestion service | `ingestion/` (API) |
| Database storage | `db/schema.sql` — Postgres |
| Multi-provider support | `sdk/inferlog/providers/` (openai, anthropic, mock) |
| Streaming responses | SSE, gateway → browser |
| Latency / throughput / error dashboards | `web/` Dashboard tab |
| One-command setup | `docker compose up` |
| Event-based architecture | Redis Streams between API and worker |
| PII redaction | `ingestion/app/redaction.py` |
| Cancel / list / resume conversations | Chat tab |

The Kubernetes deployment bonus is intentionally not included.

---

## Architecture

```
                          ┌───────────────────────────┐
        browser  ◀────────│        web  (nginx)        │   SPA + reverse proxy
                          └─────┬───────────────┬──────┘
              /api/gateway      │               │   /api/ingestion
                          ┌─────▼──────┐   ┌─────▼───────────┐
                          │  gateway   │   │  ingestion-api  │
                          │ (FastAPI)  │   │   (FastAPI)     │
                          └──┬──────┬──┘   └───┬──────────┬──┘
            conversations &  │      │ inference│ publish  │ read
            messages (sync)  │      │ events   │ event    │ aggregates
                             │      │ (SDK,    │          │
                             │      │  async)  │          │
                          ┌──▼──────▼──┐   ┌───▼────┐  ┌──▼──────────┐
                          │  Postgres  │   │ Redis  │  │  Postgres   │
                          │ conversations  │ Stream │  │ inference_  │
                          │ + messages │   └───┬────┘  │ logs        │
                          └────────────┘       │       └──▲──────────┘
                                  ▲            │ consume   │ upsert
                                  │            │ (group)   │
                                  │      ┌─────▼───────────┴──┐
                                  │      │  ingestion-worker  │
                                  └──────┤  redact + enrich   │
                                         └────────────────────┘
```

Two data paths, deliberately decoupled:

1. **Chat path (synchronous).** The gateway owns conversations and
   messages. It calls the model through the `inferlog` SDK and streams
   tokens straight back to the browser over SSE. This path never waits on
   the logging path.
2. **Logging path (asynchronous).** The SDK times the call and hands an
   event to a background dispatcher, which batches and POSTs to the
   ingestion API. The API validates and drops it on a Redis stream; the
   worker consumes, redacts PII, enriches, and upserts it. The dashboard
   reads aggregates back out.

If the logging path is slow or down, chat is unaffected — the SDK retries
in the background and sheds load if it has to.

A longer write-up — ingestion flow, failure handling, scaling — is in
[`docs/architecture.md`](docs/architecture.md).

### Using the SDK in your own app

The SDK is designed to be the thing a customer drops into their own
backend. Two integration paths:

**1. Auto-instrumentation (one line, recommended).**

```python
import inferlog
inferlog.init(
    service="my-app",
    endpoint="https://ingest.example.com/v1/ingest",
    api_key="...",
)

# Your existing code is unchanged — auto-captured:
resp = await openai_client.chat.completions.create(model="gpt-4o-mini", ...)

# Optional: tag a scope so the events carry context
with inferlog.context(conversation_id=cid, user_id=uid, tenant_id=tid):
    await openai_client.chat.completions.create(...)
```

`init()` returns the list of providers it actually instrumented
(`openai`, `anthropic`); auto-instrumentation only patches libraries that
are importable in your process.

**2. Explicit wrapper.** For custom or in-house providers there's
`LoggedLLMClient`. It shares the same dispatcher + redactor, so events
look identical regardless of which path produced them.

The SDK ships with the production essentials you'd expect:

| Concern | What you get |
|---|---|
| **PII redaction** | Default regex pass (email, phone, card, SSN, IP, API key); extensible (`extra_patterns=…`); replaceable (`custom=…` callable for Presidio / NER / your own classifier) |
| **Sampling** | `KeepAll` (default), `Probability(rate)`, `AlwaysKeepErrors(inner)`, `CustomSampler(fn)` |
| **Backpressure** | Bounded queue with `on_drop(count, reason)` callback; reasons: `queue_full`, `max_retries`, `permanent_error`, `sampled` |
| **Network resilience** | Retry with exponential backoff + jitter; honours `Retry-After` on 429 / 503 |
| **Auth** | `auth_scheme="x-api-key"` (default) or `"bearer"` |
| **Shutdown** | `atexit` flush on process exit; `await inferlog.ashutdown()` for clean async drain |
| **Observability of the logger itself** | `inferlog.stats()` returns queue depth, delivered, dropped, closed |
| **Safety** | Every wrapper is in `try/except` — the SDK will not break your call path |

### Repo layout

```
sdk/         inferlog — the logging SDK (provider adapters, dispatcher, client)
gateway/     chat backend: conversations, multi-turn context, SSE streaming
ingestion/   ingestion API + stream worker (one image, two entrypoints)
web/         React UI — chat + dashboards
db/schema.sql   the shared Postgres schema
scripts/seed.py synthetic data generator for the dashboard
```

---

## Schema design

Three tables (`db/schema.sql`). The guiding decision is **who owns what**:
the gateway owns conversation state, the worker owns telemetry, and they
are coupled as loosely as a single database allows.

**`conversations`** — one row per chat. Holds the provider/model so a
conversation is reproducible, and a `system_prompt`.

**`messages`** — one row per turn, FK to `conversations` with
`ON DELETE CASCADE`. `role` is checked (`system`/`user`/`assistant`).
`status` (`complete`/`cancelled`/`error`) lets the UI render a half-finished
answer honestly after a cancel. `request_id` links a turn to its inference
log.

**`inference_logs`** — one row per model call. Notable choices:

- **`request_id` is `UNIQUE`** and generated by the SDK. It's the
  idempotency key: the worker upserts `ON CONFLICT (request_id) DO NOTHING`,
  so at-least-once stream delivery can't create duplicates.
- **No foreign key to `conversations`.** Logs are an independent telemetry
  stream. They must survive a deleted conversation, and the worker (which
  may run ahead of, or behind, the gateway) shouldn't fail a write because
  a conversation row isn't there yet. `conversation_id` is a soft reference
  for joins.
- **Enriched fields are stored, not computed on read** — `total_tokens`,
  `tokens_per_second`, `estimated_cost_usd`, `pii_redaction_count`. The
  dashboards are read-heavy and refresh on a timer; doing the math once at
  ingest is cheaper than on every query.
- **Three indexes** (`started_at`, `status`, `conversation_id`) — exactly
  what the metric queries filter and sort on.

Previews are capped at ~280 characters. We store a redacted preview, never
the full prompt or completion, in the telemetry table.

---

## Tradeoffs

Things I'd do differently with more time, or chose deliberately:

- **One Postgres, two owners.** A purist would give the telemetry store its
  own database (or a columnar store like ClickHouse). For this scope, one
  Postgres with clear ownership boundaries is far simpler to run and
  reason about. The no-FK decision keeps the coupling honest.
- **PII redaction is regex-based and best-effort.** It reliably catches
  emails, phone numbers, cards, SSNs, IPs and API-key-shaped strings. It
  will miss names, addresses, and anything novel. The SDK exposes a
  pluggable interface — wire up Microsoft Presidio, spaCy NER, or an LLM
  judge for richer coverage. A fast, predictable pass on every event is
  still the right *first* line of defence.
- **Redaction runs inside the SDK, in the host process.** Raw PII never
  crosses the wire. The ingestion worker keeps an opt-in defense-in-depth
  pass (`INGEST_DEFENSE_IN_DEPTH_REDACT=true`) for legacy or non-SDK
  posters, but it's off by default — the SDK owns the contract.
- **The worker DLQs and continues on a processing error** rather than
  retrying transient failures. It keeps one bad event from wedging the
  stream, and the DLQ is the audit trail — but a genuinely transient DB
  blip currently lands a recoverable event in the DLQ. A retry-with-budget
  before DLQ would be the next iteration.
- **Test dependencies ship in the service images.** `pytest` is in each
  service's `requirements.txt` so `make test` is a one-liner. For a real
  deployment that belongs in a separate build stage.
- **Timeseries buckets with no events are simply absent** from the API
  response (no zero-fill). Fine for a live dashboard; a report would want
  `generate_series` to fill the gaps.
- **Conversation context is a plain sliding window** of the last N
  messages — no summarisation or token-budget trimming. It's the
  "short conversational context" the brief asked for; longer memory is a
  separate feature.

## What I'd improve with more time

- A retry budget in the worker before the DLQ, plus a DLQ replay command.
- Move the dashboard reads off the write database (read replica, or a
  rollup table refreshed on a schedule) once log volume is real.
- Per-`conversation_id` log views in the UI, joining `messages` to
  `inference_logs` so you can see the cost/latency of a specific chat.
- Auth on the whole thing — right now only the SDK→ingestion hop is
  authenticated (a shared key); the browser-facing APIs are open.
- Replace the single-key ingestion auth with per-service credentials.
- A proper migration tool (Alembic) instead of an idempotent `schema.sql`.

---

## Configuration

Everything has a working default; `.env` is optional. The keys that matter:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | enables the GPT models |
| `ANTHROPIC_API_KEY` | _(empty)_ | enables the Claude model |
| `INGEST_API_KEY` | `local-dev-ingest-key` | shared secret, SDK → ingestion API |
| `CONTEXT_WINDOW_MESSAGES` | `12` | how many recent messages to replay as context |

Ports: web **8088**, gateway **8086**, ingestion API **8081**, Postgres
**5432**, Redis **6379**. (The gateway uses host port 8086 because 8080 is a
common local collision; inside the network it's still 8080.)

## Testing

```bash
make test
```

Brings up Postgres + Redis and runs three suites inside the service images:

- **SDK** (13) — dispatcher batching/retry/drop, the client wrapper,
  error classification, cancellation logging.
- **Gateway** (10) — conversation CRUD, streaming, multi-turn context,
  cancellation persisting a partial turn.
- **Ingestion** (25) — redaction, enrichment, the ingest API, and the
  worker (storage, idempotency, PII, DLQ).

The suites use the offline mock provider, so they need no API keys and make
no external calls.

## API, briefly

Gateway (`/api/gateway` via the web proxy, or `:8086` directly):

```
GET    /v1/models
POST   /v1/conversations                      {model}
GET    /v1/conversations
GET    /v1/conversations/{id}                 full history (resume)
DELETE /v1/conversations/{id}
POST   /v1/conversations/{id}/messages        {content}  → SSE stream
```

Ingestion (`/api/ingestion`, or `:8081`):

```
POST   /v1/ingest                  batch of events  (x-api-key required)
GET    /v1/metrics/summary?window=
GET    /v1/metrics/timeseries?window=&bucket=
GET    /v1/metrics/errors?window=
GET    /v1/logs?limit=&status=&conversation_id=
```
