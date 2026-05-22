import pytest

from tests.conftest import create_conversation


async def test_models_endpoint_lists_mock(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    models = {m["model"] for m in resp.json()}
    assert "mock-1" in models


async def test_create_and_list_conversation(client):
    cid = await create_conversation(client)

    resp = await client.get("/v1/conversations")
    assert resp.status_code == 200
    listed = resp.json()
    assert len(listed) == 1
    assert listed[0]["id"] == cid
    assert listed[0]["message_count"] == 0


async def test_create_with_unavailable_model_is_rejected(client):
    resp = await client.post("/v1/conversations", json={"model": "gpt-4.1"})
    # No OpenAI key in tests → that model is not on offer.
    assert resp.status_code == 400


async def test_get_missing_conversation_returns_404(client):
    resp = await client.get("/v1/conversations/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


async def test_delete_conversation(client):
    cid = await create_conversation(client)
    assert (await client.delete(f"/v1/conversations/{cid}")).status_code == 204
    assert (await client.get(f"/v1/conversations/{cid}")).status_code == 404
