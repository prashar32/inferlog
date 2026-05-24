"""Ingestion-side validation of the inference event.

This mirrors the SDK's InferenceEvent (sdk/inferlog/events.py). It is a
separate model on purpose: the ingestion service should not trust the
producer blindly, and re-validating here is the contract boundary.
`extra="ignore"` means a newer SDK can add fields without breaking us.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["success", "error", "cancelled"]


class IngestEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: UUID
    service: str
    provider: str
    model: str
    status: Status
    streamed: bool
    started_at: datetime
    completed_at: datetime
    latency_ms: int = Field(ge=0)

    conversation_id: UUID | None = None
    ttft_ms: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    # Previews arrive ALREADY REDACTED — the SDK does that in-process so
    # raw PII never crosses the wire. `pii_redaction_count` tells us how
    # many substitutions happened.
    input_preview: str | None = None
    output_preview: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    pii_redaction_count: int = Field(default=0, ge=0)
    # Free-form tags from `inferlog.context(...)` on the SDK side.
    tags: dict = Field(default_factory=dict)
    client_metadata: dict = Field(default_factory=dict)

    sdk_version: str | None = None
    schema_version: int = 1
    # Stamped by the ingestion API when the event is accepted, not by the SDK.
    received_at: datetime | None = None


class IngestBatch(BaseModel):
    events: list[IngestEvent] = Field(min_length=1, max_length=500)
