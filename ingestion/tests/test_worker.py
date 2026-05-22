from app.worker import _handle

from tests.conftest import event_payload


async def _drain(stream, db):
    """Read everything currently on the stream and run it through the worker."""
    messages = await stream.read("test-consumer", count=50, block_ms=100)
    await _handle(stream, db, messages)
    return messages


async def test_worker_stores_event_in_db(stream, db):
    payload = event_payload()
    await stream.publish(payload)

    await _drain(stream, db)

    logs = await db.recent_logs(10, None, None)
    assert len(logs) == 1
    assert str(logs[0]["request_id"]) == payload["request_id"]
    assert logs[0]["total_tokens"] == 120


async def test_worker_is_idempotent_on_redelivery(stream, db):
    payload = event_payload()
    # same request_id delivered twice — at-least-once delivery in action
    await stream.publish(payload)
    await stream.publish(payload)

    await _drain(stream, db)

    logs = await db.recent_logs(10, None, None)
    assert len(logs) == 1


async def test_worker_redacts_pii_before_storing(stream, db):
    await stream.publish(event_payload(input_preview="reach me at x@y.com"))

    await _drain(stream, db)

    logs = await db.recent_logs(10, None, None)
    assert "x@y.com" not in logs[0]["input_preview"]
    assert logs[0]["pii_redaction_count"] == 1


async def test_worker_sends_malformed_event_to_dlq(stream, db):
    await stream.publish({"this": "is not a valid event"})

    await _drain(stream, db)

    # nothing stored, but the bad event is parked in the DLQ
    assert await db.recent_logs(10, None, None) == []
    assert await stream._redis.xlen(stream._dlq) == 1


async def test_metrics_summary_reflects_stored_events(stream, db):
    await stream.publish(event_payload(status="success"))
    await stream.publish(event_payload(status="error", error_type="rate_limit"))
    await _drain(stream, db)

    summary = await db.metrics_summary(60)
    assert summary["total_requests"] == 2
    assert summary["errors"] == 1

    errors = await db.metrics_errors(60)
    assert errors["by_type"][0]["error_type"] == "rate_limit"
