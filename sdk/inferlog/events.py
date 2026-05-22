"""The inference event — the single payload shape the SDK emits.

This is the contract between the SDK and the ingestion service. The
ingestion side re-validates it with Pydantic (see ingestion/app/events.py);
if you change a field here, change it there too. `schema_version` exists so
that mismatch is at least detectable rather than silent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

SDK_VERSION = "0.1.0"
SCHEMA_VERSION = 1


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class InferenceEvent:
    request_id: str            # SDK-generated UUID; also the ingestion idempotency key
    service: str               # logical emitter, e.g. "chat-gateway"
    provider: str
    model: str
    status: str                # "success" | "error" | "cancelled"
    streamed: bool
    started_at: datetime
    completed_at: datetime
    latency_ms: int

    conversation_id: str | None = None
    ttft_ms: int | None = None          # time to first token, streaming only
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    input_preview: str | None = None    # raw here; the worker redacts PII
    output_preview: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    client_metadata: dict = field(default_factory=dict)

    sdk_version: str = SDK_VERSION
    schema_version: int = SCHEMA_VERSION

    def to_payload(self) -> dict:
        """JSON-ready dict for the ingestion endpoint."""
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["completed_at"] = self.completed_at.isoformat()
        return data
