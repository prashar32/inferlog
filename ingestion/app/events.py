"""Ingestion-side validation of the inference event.

This mirrors the SDK's InferenceEvent (sdk/inferlog/events.py). It is a
separate model on purpose: the ingestion service should not trust the
producer blindly, and re-validating here is the contract boundary.
`extra="ignore"` means a newer SDK can add fields without breaking us.

Per-field length caps are enforced server-side as defense against a
buggy or hostile client. The SDK caps the same fields on its side at
`to_payload()` — we re-check here because we are NOT obliged to trust
the SDK version a particular customer is running.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["success", "error", "cancelled"]

# Server-side caps. Match the SDK's defaults in sdk/inferlog/events.py;
# previews are bigger here because the SDK truncates to 280 chars but a
# future SDK version may carry more.
_MAX_PREVIEW_CHARS = 4000
_MAX_ERROR_CHARS = 2000
_MAX_ERROR_TYPE_CHARS = 64
_MAX_NAME_CHARS = 128            # service / provider / model
_MAX_DICT_BYTES = 8192           # tags / client_metadata serialized JSON


def _bounded_dict(value: dict | None, field_name: str) -> dict:
    if not value:
        return {}
    encoded = json.dumps(value, default=str).encode("utf-8")
    if len(encoded) > _MAX_DICT_BYTES:
        raise ValueError(
            f"{field_name} exceeds {_MAX_DICT_BYTES} bytes when serialized"
        )
    return value


class IngestEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: UUID
    service: str = Field(max_length=_MAX_NAME_CHARS)
    provider: str = Field(max_length=_MAX_NAME_CHARS)
    model: str = Field(max_length=_MAX_NAME_CHARS)
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
    input_preview: str | None = Field(default=None, max_length=_MAX_PREVIEW_CHARS)
    output_preview: str | None = Field(default=None, max_length=_MAX_PREVIEW_CHARS)
    error_type: str | None = Field(default=None, max_length=_MAX_ERROR_TYPE_CHARS)
    error_message: str | None = Field(default=None, max_length=_MAX_ERROR_CHARS)
    pii_redaction_count: int = Field(default=0, ge=0)
    # Free-form tags from `inferlog.context(...)` on the SDK side.
    tags: dict = Field(default_factory=dict)
    client_metadata: dict = Field(default_factory=dict)

    sdk_version: str | None = Field(default=None, max_length=_MAX_NAME_CHARS)
    schema_version: int = 1
    # Stamped by the ingestion API when the event is accepted, not by the SDK.
    received_at: datetime | None = None

    @field_validator("tags")
    @classmethod
    def _cap_tags(cls, v):
        return _bounded_dict(v, "tags")

    @field_validator("client_metadata")
    @classmethod
    def _cap_metadata(cls, v):
        return _bounded_dict(v, "client_metadata")


class IngestBatch(BaseModel):
    events: list[IngestEvent] = Field(min_length=1, max_length=500)
