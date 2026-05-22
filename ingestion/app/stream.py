"""Redis Streams wrapper — the event bus between the ingestion API and the
worker.

Why Redis Streams (and not pub/sub, or Kafka): it's a durable, replayable
log with consumer groups and per-message acks, which is exactly the
at-least-once delivery this needs — but it's a single small container, not
an operational commitment. Each event is stored as one JSON blob under a
`data` field.
"""

from __future__ import annotations

import json
import logging

import redis.asyncio as redis
from redis.exceptions import ResponseError

log = logging.getLogger("ingestion.stream")

Message = tuple[str, dict]  # (stream id, decoded event)


class EventStream:
    def __init__(self, url: str, *, stream_key: str, group: str, dlq_key: str):
        self._redis = redis.from_url(url, decode_responses=True)
        self._stream = stream_key
        self._group = group
        self._dlq = dlq_key

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001
            return False

    async def ensure_group(self) -> None:
        """Create the consumer group, tolerating that it may already exist."""
        try:
            await self._redis.xgroup_create(
                self._stream, self._group, id="0", mkstream=True
            )
            log.info("created consumer group '%s' on '%s'", self._group, self._stream)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def publish(self, event: dict) -> str:
        return await self._redis.xadd(
            self._stream, {"data": json.dumps(event, default=str)}
        )

    @staticmethod
    def _decode(entries) -> list[Message]:
        decoded: list[Message] = []
        for msg_id, fields in entries or []:
            raw = fields.get("data", "")
            try:
                decoded.append((msg_id, json.loads(raw)))
            except json.JSONDecodeError:
                # Surfaces as a DLQ entry once the worker tries to validate it.
                decoded.append((msg_id, {"__unparseable__": raw}))
        return decoded

    async def read(self, consumer: str, count: int, block_ms: int) -> list[Message]:
        """Block-read new (never-delivered) messages for this consumer."""
        resp = await self._redis.xreadgroup(
            self._group, consumer, {self._stream: ">"}, count=count, block=block_ms
        )
        if not resp:
            return []
        _, entries = resp[0]
        return self._decode(entries)

    async def claim_stale(
        self, consumer: str, min_idle_ms: int, count: int
    ) -> list[Message]:
        """Reclaim messages another consumer took but never acked (crashed)."""
        result = await self._redis.xautoclaim(
            self._stream, self._group, consumer, min_idle_ms,
            start_id="0-0", count=count,
        )
        # redis 7 returns [cursor, entries, deleted]; 6.2 returns [cursor, entries].
        entries = result[1] if len(result) > 1 else []
        return self._decode(entries)

    async def ack(self, msg_id: str) -> None:
        await self._redis.xack(self._stream, self._group, msg_id)

    async def to_dlq(self, raw: dict, error: str) -> None:
        """Park a message we cannot process so the main stream keeps flowing."""
        await self._redis.xadd(
            self._dlq,
            {"error": str(error)[:1000], "data": json.dumps(raw, default=str)},
        )
        log.warning("event sent to DLQ: %s", str(error)[:200])

    async def depth(self) -> int:
        return await self._redis.xlen(self._stream)

    async def pending_count(self) -> int:
        try:
            info = await self._redis.xpending(self._stream, self._group)
        except ResponseError:
            return 0
        return info["pending"] if isinstance(info, dict) else int(info[0] or 0)

    async def close(self) -> None:
        await self._redis.aclose()
