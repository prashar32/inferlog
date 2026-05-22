from app.events import IngestEvent
from app.processing import process_event

from tests.conftest import event_payload


def _event(**overrides) -> IngestEvent:
    return IngestEvent.model_validate(event_payload(**overrides))


def test_process_redacts_previews_and_counts():
    record = process_event(
        _event(input_preview="my email is a@b.com", output_preview="noted")
    )
    assert "[REDACTED_EMAIL]" in record["input_preview"]
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
