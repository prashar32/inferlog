"""Turn a validated IngestEvent into a row ready for `inference_logs`.

Previews arrive PRE-REDACTED from the SDK — that is the point of doing PII
redaction in the customer's process, before the event ever crosses the
wire. Here we only do enrichment (token totals, throughput, cost) and a
defense-in-depth redaction pass behind a feature flag.

Kept pure (no I/O) so it's trivial to unit-test.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from . import enrich
from .events import IngestEvent
from .redaction import redact as defense_in_depth_redact

# Off by default — the SDK already redacted. Turn on if you don't fully
# trust the SDK version a particular client is on. The count is additive.
_DEFENSE_IN_DEPTH = os.getenv("INGEST_DEFENSE_IN_DEPTH_REDACT", "false").lower() == "true"


def process_event(event: IngestEvent) -> dict:
    input_preview = event.input_preview
    output_preview = event.output_preview
    error_message = event.error_message
    extra_redactions = 0

    if _DEFENSE_IN_DEPTH:
        # Server-side second pass — useful if a non-SDK source posts to
        # /v1/ingest or you suspect an older SDK on a customer.
        input_preview, n1 = defense_in_depth_redact(input_preview)
        output_preview, n2 = defense_in_depth_redact(output_preview)
        error_message, n3 = defense_in_depth_redact(error_message)
        extra_redactions = n1 + n2 + n3

    return {
        "request_id": event.request_id,
        "conversation_id": event.conversation_id,
        "service": event.service,
        "provider": event.provider,
        "model": event.model,
        "status": event.status,
        "streamed": event.streamed,
        "started_at": event.started_at,
        "completed_at": event.completed_at,
        "latency_ms": event.latency_ms,
        "ttft_ms": event.ttft_ms,
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "total_tokens": enrich.total_tokens(
            event.prompt_tokens, event.completion_tokens, event.total_tokens
        ),
        "tokens_per_second": enrich.tokens_per_second(
            event.completion_tokens, event.latency_ms
        ),
        "estimated_cost_usd": enrich.estimate_cost(
            event.model, event.prompt_tokens, event.completion_tokens
        ),
        "error_type": event.error_type,
        "error_message": error_message,
        "input_preview": input_preview,
        "output_preview": output_preview,
        "pii_redaction_count": event.pii_redaction_count + extra_redactions,
        "sdk_version": event.sdk_version,
        "schema_version": event.schema_version,
        # Merge SDK-side tags into client_metadata for storage convenience.
        "client_metadata": {**event.client_metadata, **(event.tags or {})},
        "received_at": event.received_at or datetime.now(timezone.utc),
    }
