# Architecture notes

Companion to the README. Covers the four things the brief asked for —
ingestion flow, logging strategy, scaling, failure handling — plus the
SDK contract a customer integrates against.

## The product picture

The SDK lives **inside the customer's application**, not inside our
infrastructure. That single fact dictates two design rules:

1. **PII redaction happens in the customer's process.** Doing it
   server-side would mean raw PII has already crossed the network — the
   whole point of redaction is that the sensitive bytes never leave the
   boundary they were created in. The default regex pass runs inline in
   the SDK; customers can add patterns or plug in stronger redactors
   (Presidio, NER, an LLM judge). The ingestion worker keeps an opt-in
   defense-in-depth pass for legacy or non-SDK posters.

2. **Integration is one line, truly model-agnostic.** Customers call
   `inferlog.init(...)` at process startup. The SDK patches
   `httpx.AsyncClient.send` / `httpx.Client.send` — the universal
   async transport every modern Python LLM library uses. Any LLM HTTP
   request — through the OpenAI SDK, Anthropic SDK, raw `httpx` calls
   to a self-hosted vLLM / Ollama, an OpenAI-compatible proxy, or
   LangChain / LlamaIndex on top of any of those — is captured. URL
   patterns identify the provider; per-provider parsers extract model,
   tokens, output text. There's also an explicit `LoggedLLMClient` for
   in-process custom providers (which don't go over HTTP — the demo's
   mock model uses this path), with a contextvar that prevents the two
   paths from double-logging.

## Ingestion flow

A single inference event travels:

```
SDK (in gateway)  ──POST /v1/ingest──▶  ingestion-api
                                            │ validate (Pydantic) + stamp received_at
                                            ▼
                                       Redis stream  "inferlog:events"   (XADD)
                                            │
                                            ▼  XREADGROUP, consumer group "workers"
                                       ingestion-worker
                                            │ validate · redact PII · enrich
                                            ▼
                                       Postgres  inference_logs   (idempotent upsert)
```

1. **Produce.** The SDK builds an `InferenceEvent` the moment a model call
   finishes (success, error, or cancelled) and queues it. A background
   dispatcher batches queued events and POSTs them to `/v1/ingest`.

2. **Accept.** The ingestion API authenticates the batch (`x-api-key`),
   validates every event against a Pydantic model, stamps `received_at`,
   and `XADD`s each one to a Redis stream. It returns `202` immediately —
   it never touches Postgres. The write path is intentionally thin.

3. **Consume.** The worker reads the stream with `XREADGROUP` under a
   consumer group. For each event it: re-validates, redacts PII from the
   previews, derives metadata (token totals, tokens/sec, estimated cost),
   and upserts into `inference_logs`. It acks the message only after the
   row is committed.

4. **Read.** The dashboard hits `/v1/metrics/*`, which runs aggregate
   queries (`percentile_cont`, time-bucketing) over `inference_logs`.

The API↔worker split via a stream is the "event-based architecture": the
producer and consumer scale, deploy, and fail independently, and the stream
is a durable buffer between them.

## Logging strategy

**The logging path must never degrade the chat path.** Everything follows
from that:

- The SDK's `submit()` is synchronous, non-blocking, and cannot raise. It
  drops into a bounded in-memory queue.
- A background task batches the queue (by size or a 0.5s timer — "near
  real time") and delivers with retry + exponential backoff.
- If the queue is full (ingestion is down or slow), events are **dropped
  and counted**, not blocked on. Shedding telemetry is the correct failure
  mode; stalling a user's chat is not.
- Logging is emitted in a `finally`, so a call that errors or is cancelled
  is logged just like a successful one — with the matching `status`.

**What's captured:** model, provider, latency, time-to-first-token, token
usage, status, error type/message, timestamps, conversation id, and
truncated input/output previews. Token classification of errors into a
small stable set (`rate_limit`, `timeout`, `auth`, …) happens in the SDK so
the dashboards can group on it.

**Where PII redaction happens.** Inside the SDK, in the customer's
process, before the event reaches the dispatcher queue. Raw bytes never
cross the wire. Default is a regex pass; customers can extend (extra
patterns) or replace (a custom callable for Presidio / NER / LLM judge).
The server-side worker keeps an opt-in second pass for defense in depth
(`INGEST_DEFENSE_IN_DEPTH_REDACT=true`), off by default.

**What's stored:** already-redacted previews plus the SDK-supplied
redaction count. Enriched fields (token totals, throughput, cost) are
computed once at ingest, not on every read.

**SDK integration shape.** Two paths, deliberately:

1. **HTTP-level capture** (default). A single
   `inferlog.init(api_key=..., endpoint=...)` at startup patches
   `httpx.AsyncClient.send` / `httpx.Client.send`. Any LLM call going
   out through `httpx` — OpenAI SDK, Anthropic SDK, raw httpx calls,
   self-hosted models (vLLM, Ollama), OpenAI-compatible proxies, the
   long tail — is captured. URL pattern → provider handler →
   per-provider parser. Adding a new provider is one handler class.
   Customer's existing code is untouched. Walkthrough:
   `_notes/http-capture-walkthrough.md`.
2. **Explicit wrapper.** `LoggedLLMClient` for in-process providers
   that don't speak HTTP (mock models, custom in-house inference).
   Shares the global dispatcher and redactor; a contextvar suppresses
   HTTP capture inside this scope to avoid double-logging.

**Production-shaped SDK essentials** (built into the same package):
- Sampling — `KeepAll` / `Probability(rate)` / `AlwaysKeepErrors(inner)`
  / `CustomSampler(fn)`.
- Backpressure visibility — `on_drop(count, reason)` callback.
- Network resilience — exponential backoff with jitter; honours
  `Retry-After` on 429 / 503.
- Auth schemes — `x-api-key` (default) or `Authorization: Bearer …`.
- Shutdown — atexit flush on process exit, async `await ashutdown()` for
  clean drain.
- Self-observability — `inferlog.stats()` returns queue depth,
  delivered, dropped, closed.
- Safety — every wrapper is in `try/except`; the SDK will not break the
  host call path.

## Scaling considerations

The design scales by component, not as a monolith:

- **Ingestion API** is stateless — put N behind a load balancer. It only
  validates and `XADD`s, so it's CPU-cheap and fast.
- **Worker** scales horizontally for free: add more workers to the same
  consumer group and Redis partitions messages across them. Throughput is
  `workers × batch_size`.
- **Redis stream** absorbs bursts — it's the buffer that lets the worker
  fall behind during a spike and catch up after, without backpressure
  reaching the chat path. `XLEN`/pending depth is the lag metric to alert
  on.
- **Postgres** is the first thing that hurts under real volume. The honest
  scaling answer: (a) it carries an OLTP store and a telemetry store today
  — split them; (b) move dashboard reads to a read replica or a rollup
  table; (c) at serious volume, `inference_logs` belongs in a columnar /
  time-series store (ClickHouse, Timescale). The schema is already
  append-only and time-ordered, which makes that migration straightforward.
- **Gateway** is stateless except for the DB; it scales horizontally. SSE
  streams are long-lived connections, so connection count is the limit to
  watch, not CPU.

Batching is the main throughput lever already in place: the SDK batches up
to 25 events per request, and the worker reads up to 50 per loop.

## Failure handling assumptions

- **Delivery is at-least-once.** The worker acks only after a successful
  commit. A crash between commit and ack means redelivery — which is safe,
  because the upsert is idempotent on `request_id`.
- **Orphaned messages are reclaimed.** If a worker dies holding unacked
  messages, another worker's `XAUTOCLAIM` sweep picks them up after an idle
  timeout. No message is stranded by a crash.
- **Poison messages go to a DLQ.** An event that fails validation or
  processing is moved to `inferlog:events:dlq` with the error, then acked,
  so one bad event cannot wedge the stream. The DLQ is the audit trail.
- **Telemetry loss is acceptable; chat failure is not.** If ingestion is
  fully down, the SDK retries, then drops with a counter. Chat keeps
  working. This is a deliberate, asymmetric choice.
- **The model call itself can fail.** Provider errors are caught, classified,
  surfaced to the user as a clean error event, *and* logged. Cancellation
  (the Stop button) is treated as a first-class outcome: the stream stops,
  the partial answer is persisted, and the call is logged as `cancelled`.
- **Startup ordering is not assumed.** Every service applies the schema
  idempotently on boot and retries its connections, so "worker started
  before Postgres was ready" is a non-event.