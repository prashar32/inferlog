"""InferenceEvent.to_payload() enforces hard upper bounds on free-form
fields. A buggy call site cannot ship multi-MB events that DoS our
ingestion side; the cap is silent (telemetry must not raise) but the
payload that leaves the SDK is always bounded.
"""

from __future__ import annotations

import json

from inferlog.events import (
    _MAX_DICT_BYTES,
    _MAX_ERROR_CHARS,
    _MAX_TAG_KEYS,
    _MAX_VALUE_CHARS,
)
from tests.conftest import make_event


def test_error_message_truncated_to_cap():
    event = make_event(status="error", error_message="x" * (_MAX_ERROR_CHARS * 2))
    payload = event.to_payload()
    assert len(payload["error_message"]) == _MAX_ERROR_CHARS
    # Last char is the truncation marker — easy to spot in logs.
    assert payload["error_message"].endswith("…")


def test_short_error_message_passes_through():
    event = make_event(status="error", error_message="boom")
    payload = event.to_payload()
    assert payload["error_message"] == "boom"


def test_tags_value_length_capped():
    long_value = "y" * (_MAX_VALUE_CHARS * 4)
    event = make_event(tags={"user_id": long_value})
    payload = event.to_payload()
    assert len(payload["tags"]["user_id"]) == _MAX_VALUE_CHARS


def test_tags_key_count_capped():
    too_many = {f"k{i}": str(i) for i in range(_MAX_TAG_KEYS * 3)}
    event = make_event(tags=too_many)
    payload = event.to_payload()
    assert len(payload["tags"]) <= _MAX_TAG_KEYS


def test_tags_total_bytes_capped():
    big = {f"k{i}": "v" * _MAX_VALUE_CHARS for i in range(_MAX_TAG_KEYS)}
    event = make_event(tags=big)
    payload = event.to_payload()
    assert len(json.dumps(payload["tags"]).encode("utf-8")) <= _MAX_DICT_BYTES


def test_client_metadata_capped_too():
    """tags and client_metadata are different fields but share the cap rule."""
    event = make_event(client_metadata={"big": "z" * (_MAX_VALUE_CHARS * 10)})
    payload = event.to_payload()
    assert len(payload["client_metadata"]["big"]) == _MAX_VALUE_CHARS


def test_empty_dicts_serialise_as_empty():
    event = make_event(tags={}, client_metadata={})
    payload = event.to_payload()
    assert payload["tags"] == {}
    assert payload["client_metadata"] == {}


def test_caps_never_raise():
    """Telemetry must not break the host call path even with garbage input."""
    weird = {1: object(), "ok": "fine"}  # non-string key, non-stringable value
    event = make_event(tags=weird)
    payload = event.to_payload()
    # Whatever made it through was stringified; no exception.
    assert isinstance(payload["tags"], dict)
