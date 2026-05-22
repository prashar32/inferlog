from tests.conftest import API_KEY, event_payload


async def test_ingest_rejects_missing_api_key(client):
    resp = await client.post("/v1/ingest", json={"events": [event_payload()]})
    assert resp.status_code == 401


async def test_ingest_accepts_batch_and_publishes_to_stream(client, stream):
    body = {"events": [event_payload(), event_payload()]}
    resp = await client.post("/v1/ingest", headers={"x-api-key": API_KEY}, json=body)
    assert resp.status_code == 202
    assert resp.json()["accepted"] == 2
    # the events really landed on the bus
    assert await stream.depth() == 2


async def test_ingest_rejects_malformed_event(client):
    resp = await client.post(
        "/v1/ingest",
        headers={"x-api-key": API_KEY},
        json={"events": [{"request_id": "not-a-uuid"}]},
    )
    assert resp.status_code == 422


async def test_metrics_summary_on_empty_db(client):
    resp = await client.get("/v1/metrics/summary?window=60")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] == 0
    assert body["by_model"] == []


async def test_healthz_reports_dependencies(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db"] is True
    assert body["redis"] is True
