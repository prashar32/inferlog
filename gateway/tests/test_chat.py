import asyncio

import pytest

from tests.conftest import create_conversation, parse_sse


async def test_send_message_streams_and_persists(client, sink):
    cid = await create_conversation(client)
    resp = await client.post(
        f"/v1/conversations/{cid}/messages", json={"content": "hello there"}
    )
    assert resp.status_code == 200

    events = parse_sse(resp.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start"
    assert "token" in kinds
    assert kinds[-1] == "done"

    detail = (await client.get(f"/v1/conversations/{cid}")).json()
    roles = [(m["role"], m["status"]) for m in detail["messages"]]
    assert roles == [("user", "complete"), ("assistant", "complete")]

    await asyncio.sleep(0.15)  # let the log dispatcher flush
    assert len(sink.events) == 1
    log = sink.events[0]
    assert log["status"] == "success"
    assert log["streamed"] is True
    assert log["conversation_id"] == cid
    assert log["ttft_ms"] is not None


async def test_multi_turn_keeps_context(client, sink):
    """A second turn should carry more prompt tokens than the first — proof
    the sliding-window history is actually being replayed to the model."""
    cid = await create_conversation(client)
    await client.post(f"/v1/conversations/{cid}/messages", json={"content": "first message"})
    await client.post(f"/v1/conversations/{cid}/messages", json={"content": "second message"})

    detail = (await client.get(f"/v1/conversations/{cid}")).json()
    assert [m["role"] for m in detail["messages"]] == [
        "user", "assistant", "user", "assistant",
    ]

    await asyncio.sleep(0.15)
    assert len(sink.events) == 2
    assert sink.events[1]["prompt_tokens"] > sink.events[0]["prompt_tokens"]


async def test_resume_returns_full_history(client):
    cid = await create_conversation(client)
    await client.post(f"/v1/conversations/{cid}/messages", json={"content": "remember 42"})

    # "Resuming" is just fetching the conversation again.
    detail = (await client.get(f"/v1/conversations/{cid}")).json()
    assert len(detail["messages"]) == 2
    assert detail["title"] == "remember 42"  # title auto-set from first message


async def test_cancel_stops_generation_and_logs_it(client, sink):
    cid = await create_conversation(client)

    request = asyncio.create_task(
        client.post(
            f"/v1/conversations/{cid}/messages",
            json={"content": "give me a long answer please"},
        )
    )
    await asyncio.sleep(0.08)  # let a few tokens stream
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    await asyncio.sleep(0.5)  # let cancellation + background persist settle

    cancelled_logs = [e for e in sink.events if e["status"] == "cancelled"]
    assert len(cancelled_logs) == 1

    detail = (await client.get(f"/v1/conversations/{cid}")).json()
    assistant = [m for m in detail["messages"] if m["role"] == "assistant"]
    # Either a partial assistant turn marked 'cancelled', or none if we
    # cancelled before the first token — both are valid, neither is 'complete'.
    assert all(m["status"] == "cancelled" for m in assistant)


async def test_send_to_missing_conversation_returns_404(client):
    resp = await client.post(
        "/v1/conversations/00000000-0000-0000-0000-000000000000/messages",
        json={"content": "hi"},
    )
    assert resp.status_code == 404
