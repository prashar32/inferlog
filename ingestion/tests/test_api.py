from tests.conftest import API_KEY, event_payload


AUTH = {"x-api-key": API_KEY}


async def test_ingest_rejects_missing_api_key(client):
    resp = await client.post("/v1/ingest", json={"events": [event_payload()]})
    assert resp.status_code == 401


async def test_ingest_accepts_bearer_auth(client, stream):
    """The SDK can be configured with `auth_scheme="bearer"`; the API must
    accept both `x-api-key` and `Authorization: Bearer …`."""
    resp = await client.post(
        "/v1/ingest",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"events": [event_payload()]},
    )
    assert resp.status_code == 202
    assert resp.json()["accepted"] == 1


async def test_ingest_rejects_wrong_api_key(client):
    resp = await client.post(
        "/v1/ingest",
        headers={"x-api-key": "wrong-key"},
        json={"events": [event_payload()]},
    )
    assert resp.status_code == 401


async def test_ingest_accepts_batch_and_publishes_to_stream(client, stream):
    body = {"events": [event_payload(), event_payload()]}
    resp = await client.post("/v1/ingest", headers=AUTH, json=body)
    assert resp.status_code == 202
    assert resp.json()["accepted"] == 2
    # the events really landed on the bus
    assert await stream.depth() == 2


async def test_ingest_rejects_malformed_event(client):
    resp = await client.post(
        "/v1/ingest",
        headers=AUTH,
        json={"events": [{"request_id": "not-a-uuid"}]},
    )
    assert resp.status_code == 422


async def test_ingest_rejects_oversized_preview(client):
    """Per-field length cap defends against a buggy / hostile client filling
    the queue with multi-MB events."""
    huge = "x" * 5000  # > 4000-char cap
    resp = await client.post(
        "/v1/ingest",
        headers=AUTH,
        json={"events": [event_payload(input_preview=huge)]},
    )
    assert resp.status_code == 422


async def test_ingest_rejects_oversized_tags(client):
    """tags dict must serialize to < 8KB."""
    fat_tags = {f"k{i}": "v" * 200 for i in range(60)}  # ~12KB serialized
    resp = await client.post(
        "/v1/ingest",
        headers=AUTH,
        json={"events": [event_payload(tags=fat_tags)]},
    )
    assert resp.status_code == 422


async def test_metrics_summary_requires_auth(client):
    """Dashboard endpoints leak every customer's logs if open — auth required."""
    resp = await client.get("/v1/metrics/summary?window=60")
    assert resp.status_code == 401


async def test_logs_requires_auth(client):
    resp = await client.get("/v1/logs?limit=10")
    assert resp.status_code == 401


async def test_metrics_summary_on_empty_db(client):
    resp = await client.get("/v1/metrics/summary?window=60", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] == 0
    assert body["by_model"] == []


async def test_healthz_reports_dependencies(client):
    """healthz stays unauth — orchestrators need to hit it without a secret."""
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db"] is True
    assert body["redis"] is True
    # DLQ depth exposed so silent data loss is alarmable.
    assert body["dlq_depth"] == 0
