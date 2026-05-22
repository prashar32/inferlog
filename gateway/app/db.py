"""Postgres access for conversations and messages.

Hand-written SQL over asyncpg — no ORM. The gateway only owns two tables
and a handful of queries, so an ORM would be more weight than help. Every
method returns plain dicts so the rest of the app never touches asyncpg
Record objects.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


def _row(record: asyncpg.Record | None) -> dict | None:
    return dict(record) if record is not None else None


class Database:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "Database":
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def apply_schema(self, schema_sql: str) -> None:
        """Idempotent — schema.sql is all CREATE ... IF NOT EXISTS."""
        async with self._pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def ping(self) -> bool:
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT 1") == 1

    # -- conversations ---------------------------------------------------

    async def create_conversation(
        self, *, provider: str, model: str, title: str | None, system_prompt: str | None
    ) -> dict:
        record = await self._pool.fetchrow(
            """
            INSERT INTO conversations (provider, model, title, system_prompt)
            VALUES ($1, $2, $3, $4)
            RETURNING *
            """,
            provider, model, title, system_prompt,
        )
        return _row(record)  # type: ignore[return-value]

    async def list_conversations(self) -> list[dict]:
        records = await self._pool.fetch(
            """
            SELECT
                c.id, c.title, c.provider, c.model, c.created_at, c.updated_at,
                (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id)
                    AS message_count,
                (SELECT content FROM messages m
                   WHERE m.conversation_id = c.id
                   ORDER BY m.created_at DESC LIMIT 1) AS last_message
            FROM conversations c
            ORDER BY c.updated_at DESC
            """
        )
        return [dict(r) for r in records]

    async def get_conversation(self, conversation_id: UUID) -> dict | None:
        return _row(
            await self._pool.fetchrow(
                "SELECT * FROM conversations WHERE id = $1", conversation_id
            )
        )

    async def set_title_if_empty(self, conversation_id: UUID, title: str) -> None:
        await self._pool.execute(
            "UPDATE conversations SET title = $2 WHERE id = $1 AND title IS NULL",
            conversation_id, title,
        )

    async def touch_conversation(self, conversation_id: UUID) -> None:
        await self._pool.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = $1",
            conversation_id,
        )

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        result = await self._pool.execute(
            "DELETE FROM conversations WHERE id = $1", conversation_id
        )
        return result.endswith("1")  # "DELETE 1" vs "DELETE 0"

    # -- messages --------------------------------------------------------

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        *,
        status: str = "complete",
        request_id: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> dict:
        record = await self._pool.fetchrow(
            """
            INSERT INTO messages
                (conversation_id, role, content, status, request_id,
                 prompt_tokens, completion_tokens)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            conversation_id, role, content, status, request_id,
            prompt_tokens, completion_tokens,
        )
        return _row(record)  # type: ignore[return-value]

    async def get_messages(self, conversation_id: UUID) -> list[dict]:
        records = await self._pool.fetch(
            """
            SELECT id, role, content, status, request_id,
                   prompt_tokens, completion_tokens, created_at
            FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            """,
            conversation_id,
        )
        return [dict(r) for r in records]

    async def recent_messages(self, conversation_id: UUID, limit: int) -> list[dict]:
        """The last `limit` messages, oldest-first, for replay as context.

        Error turns are skipped — they carry no usable content. A cancelled
        assistant turn is kept: a partial answer is still real history.
        """
        records = await self._pool.fetch(
            """
            SELECT role, content FROM (
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = $1 AND status <> 'error'
                ORDER BY created_at DESC
                LIMIT $2
            ) recent
            ORDER BY created_at ASC
            """,
            conversation_id, limit,
        )
        return [dict(r) for r in records]
