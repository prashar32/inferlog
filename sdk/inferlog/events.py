"""The inference event — the single payload shape the SDK emits.

This is the contract between the SDK and the ingestion service. The
ingestion side re-validates it with Pydantic (see ingestion/app/events.py);
if you change a field here, change it there too. `schema_version` exists so
that mismatch is at least detectable rather than silent.

The SDK enforces hard upper bounds on a few free-form fields at the
serialization boundary (`to_payload`) — `error_message`, `tags`, and
`client_metadata`. A buggy or hostile caller can't ship multi-MB events
that DoS the ingestion side.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

SDK_VERSION = "0.4.0"
SCHEMA_VERSION = 1

# Outbound size caps. Generous so legitimate payloads aren't surprised;
# tight enough that one bad caller can't fill the queue with garbage.
_MAX_ERROR_CHARS = 2000
_MAX_TAG_KEYS = 32
_MAX_VALUE_CHARS = 256
_MAX_DICT_BYTES = 4096


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _cap_str(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _cap_value(v):
    """Bound a single value. Preserve JSON-native scalar types (bool, int,
    float, None) so downstream consumers don't see `True` arriving as the
    string `"True"`. Strings are length-capped; complex types are
    stringified and capped — strictly defensive against unusual call sites."""
    if isinstance(v, str):
        return _cap_str(v, _MAX_VALUE_CHARS)
    if v is None or isinstance(v, (bool, int, float)):
        return v
    return _cap_str(str(v), _MAX_VALUE_CHARS)


def _cap_dict(d: dict | None) -> dict:
    """Cap key count, per-value length, and total JSON size of a tag/metadata
    dict. Truncation is silent — telemetry must not raise."""
    if not d:
        return {}
    items = list(d.items())[:_MAX_TAG_KEYS]
    capped = {str(k): _cap_value(v) for k, v in items}
    # Final JSON byte cap — pop arbitrary keys until we fit.
    while capped and len(json.dumps(capped, default=str).encode("utf-8")) > _MAX_DICT_BYTES:
        capped.popitem()
    return capped


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
    # Previews are redacted INSIDE the SDK before this event leaves the
    # host process. `pii_redaction_count` is the number of substitutions
    # performed across all three of input/output/error text.
    input_preview: str | None = None
    output_preview: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    pii_redaction_count: int = 0
    # Free-form tags from `inferlog.context(...)` — conversation_id and
    # user_id, etc. Conversation_id is also lifted to the dedicated column
    # above for convenience.
    tags: dict = field(default_factory=dict)
    client_metadata: dict = field(default_factory=dict)

    sdk_version: str = SDK_VERSION
    schema_version: int = SCHEMA_VERSION

    def to_payload(self) -> dict:
        """JSON-ready dict for the ingestion endpoint. Free-form fields are
        capped here so a bug at a call site can't ship unbounded payloads."""
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["completed_at"] = self.completed_at.isoformat()
        data["error_message"] = _cap_str(data.get("error_message"), _MAX_ERROR_CHARS)
        data["tags"] = _cap_dict(data.get("tags"))
        data["client_metadata"] = _cap_dict(data.get("client_metadata"))
        return data
