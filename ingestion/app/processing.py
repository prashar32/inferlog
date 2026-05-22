"""Turn a validated IngestEvent into a row ready for `inference_logs`.

This is the "extract useful metadata" step: redact PII out of the previews,
derive token totals / throughput / cost. Kept pure (no I/O) so it's trivial
to unit-test.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import enrich
from .events import IngestEvent
from .redaction import redact


def process_event(event: IngestEvent) -> dict:
    input_preview, in_redactions = redact(event.input_preview)
    output_preview, out_redactions = redact(event.output_preview)
    # Error messages occasionally echo user input — redact them too.
    error_message, err_redactions = redact(event.error_message)

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
        "pii_redaction_count": in_redactions + out_redactions + err_redactions,
        "sdk_version": event.sdk_version,
        "schema_version": event.schema_version,
        "client_metadata": event.client_metadata,
        "received_at": event.received_at or datetime.now(timezone.utc),
    }
