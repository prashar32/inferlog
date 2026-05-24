from app.events import IngestEvent
from app.processing import process_event

from tests.conftest import event_payload


def _event(**overrides) -> IngestEvent:
    return IngestEvent.model_validate(event_payload(**overrides))


def test_process_passes_through_pre_redacted_previews():
    """Previews arrive ALREADY REDACTED from the SDK — the worker doesn't
    redact again by default. The SDK-supplied count is preserved."""
    record = process_event(
        _event(
            input_preview="my email is [REDACTED_EMAIL]",
            pii_redaction_count=1,
        )
    )
    assert record["input_preview"] == "my email is [REDACTED_EMAIL]"
    assert record["pii_redaction_count"] == 1


def test_process_derives_token_and_cost_metadata():
    record = process_event(
        _event(prompt_tokens=50, completion_tokens=100, total_tokens=None, latency_ms=2000)
    )
    assert record["total_tokens"] == 150
    assert record["tokens_per_second"] == 50.0  # 100 tokens / 2s
    assert record["estimated_cost_usd"] is not None


def test_process_defaults_received_at():
    record = process_event(_event(received_at=None))
    assert record["received_at"] is not None


def test_process_merges_tags_into_client_metadata():
    record = process_event(
        _event(
            client_metadata={"channel": "web"},
            tags={"user_id": "u-1", "tenant_id": "t-9"},
        )
    )
    assert record["client_metadata"]["channel"] == "web"
    assert record["client_metadata"]["user_id"] == "u-1"
    assert record["client_metadata"]["tenant_id"] == "t-9"
