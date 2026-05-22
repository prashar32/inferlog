"""Postgres access for the ingestion side.

The worker writes `inference_logs` (idempotent upsert); the API reads it
for the dashboards. Aggregation is pushed into SQL — Postgres is good at
`percentile_cont` and time bucketing, and it keeps payloads small.
"""

from __future__ import annotations

import json
from uuid import UUID

import asyncpg


async def _init_connection(conn: asyncpg.Connection) -> None:
    # Let us pass/receive dicts for jsonb columns directly.
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


_INSERT_COLUMNS = (
    "request_id, conversation_id, service, provider, model, status, streamed, "
    "started_at, completed_at, latency_ms, ttft_ms, prompt_tokens, "
    "completion_tokens, total_tokens, tokens_per_second, estimated_cost_usd, "
    "error_type, error_message, input_preview, output_preview, "
    "pii_redaction_count, sdk_version, schema_version, client_metadata, received_at"
)
_INSERT_FIELDS = [c.strip() for c in _INSERT_COLUMNS.split(",")]


class IngestionDB:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "IngestionDB":
        pool = await asyncpg.create_pool(
            dsn, min_size=1, max_size=10, init=_init_connection
        )
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def apply_schema(self, schema_sql: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def ping(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchval("SELECT 1") == 1
        except Exception:  # noqa: BLE001
            return False

    # -- write path ------------------------------------------------------

    async def upsert_log(self, record: dict) -> bool:
        """Insert one processed log row. Returns True if it was new.

        ON CONFLICT (request_id) DO NOTHING makes this safe to call twice for
        the same event — which is exactly what at-least-once stream delivery
        will do on retries.
        """
        placeholders = ", ".join(f"${i}" for i in range(1, len(_INSERT_FIELDS) + 1))
        values = [record[name] for name in _INSERT_FIELDS]
        result = await self._pool.execute(
            f"INSERT INTO inference_logs ({_INSERT_COLUMNS}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (request_id) DO NOTHING",
            *values,
        )
        return result.split()[-1] == "1"

    # -- read path (dashboards) -----------------------------------------

    async def metrics_summary(self, window_minutes: int) -> dict:
        row = await self._pool.fetchrow(
            """
            SELECT
                count(*)                                          AS total_requests,
                count(*) FILTER (WHERE status = 'success')         AS success,
                count(*) FILTER (WHERE status = 'error')           AS errors,
                count(*) FILTER (WHERE status = 'cancelled')       AS cancelled,
                coalesce(avg(latency_ms)::int, 0)                  AS avg_latency_ms,
                coalesce(percentile_cont(0.5)
                    WITHIN GROUP (ORDER BY latency_ms)::int, 0)    AS p50_latency_ms,
                coalesce(percentile_cont(0.95)
                    WITHIN GROUP (ORDER BY latency_ms)::int, 0)    AS p95_latency_ms,
                coalesce(percentile_cont(0.99)
                    WITHIN GROUP (ORDER BY latency_ms)::int, 0)    AS p99_latency_ms,
                coalesce(avg(ttft_ms)::int, 0)                     AS avg_ttft_ms,
                coalesce(sum(total_tokens), 0)                     AS total_tokens,
                coalesce(sum(estimated_cost_usd), 0.0)             AS total_cost_usd
            FROM inference_logs
            WHERE started_at >= now() - make_interval(mins => $1)
            """,
            window_minutes,
        )
        by_model = await self._pool.fetch(
            """
            SELECT provider, model,
                   count(*)                                   AS requests,
                   count(*) FILTER (WHERE status = 'error')    AS errors,
                   coalesce(avg(latency_ms)::int, 0)           AS avg_latency_ms,
                   coalesce(sum(total_tokens), 0)              AS tokens,
                   coalesce(sum(estimated_cost_usd), 0.0)      AS cost_usd
            FROM inference_logs
            WHERE started_at >= now() - make_interval(mins => $1)
            GROUP BY provider, model
            ORDER BY requests DESC
            """,
            window_minutes,
        )
        return {
            "window_minutes": window_minutes,
            **dict(row),
            "by_model": [dict(r) for r in by_model],
        }

    async def metrics_timeseries(
        self, window_minutes: int, bucket_seconds: int
    ) -> list[dict]:
        rows = await self._pool.fetch(
            """
            SELECT
                to_timestamp(floor(extract(epoch FROM started_at) / $2) * $2)
                                                                  AS bucket,
                count(*)                                          AS requests,
                count(*) FILTER (WHERE status = 'error')          AS errors,
                count(*) FILTER (WHERE status = 'cancelled')      AS cancelled,
                coalesce(avg(latency_ms)::int, 0)                 AS avg_latency_ms,
                coalesce(percentile_cont(0.95)
                    WITHIN GROUP (ORDER BY latency_ms)::int, 0)   AS p95_latency_ms,
                coalesce(sum(total_tokens), 0)                    AS tokens
            FROM inference_logs
            WHERE started_at >= now() - make_interval(mins => $1)
            GROUP BY bucket
            ORDER BY bucket
            """,
            window_minutes, bucket_seconds,
        )
        return [dict(r) for r in rows]

    async def metrics_errors(self, window_minutes: int) -> dict:
        by_type = await self._pool.fetch(
            """
            SELECT coalesce(error_type, 'unknown') AS error_type, count(*) AS count
            FROM inference_logs
            WHERE status = 'error'
              AND started_at >= now() - make_interval(mins => $1)
            GROUP BY error_type
            ORDER BY count DESC
            """,
            window_minutes,
        )
        recent = await self._pool.fetch(
            """
            SELECT request_id, provider, model, error_type, error_message, started_at
            FROM inference_logs
            WHERE status = 'error'
              AND started_at >= now() - make_interval(mins => $1)
            ORDER BY started_at DESC
            LIMIT 20
            """,
            window_minutes,
        )
        return {
            "window_minutes": window_minutes,
            "by_type": [dict(r) for r in by_type],
            "recent": [dict(r) for r in recent],
        }

    async def recent_logs(
        self, limit: int, status: str | None, conversation_id: UUID | None
    ) -> list[dict]:
        rows = await self._pool.fetch(
            """
            SELECT request_id, conversation_id, service, provider, model, status,
                   streamed, latency_ms, ttft_ms, prompt_tokens, completion_tokens,
                   total_tokens, tokens_per_second, estimated_cost_usd, error_type,
                   error_message, input_preview, output_preview, pii_redaction_count,
                   started_at, processed_at
            FROM inference_logs
            WHERE ($2::text IS NULL OR status = $2)
              AND ($3::uuid IS NULL OR conversation_id = $3)
            ORDER BY started_at DESC
            LIMIT $1
            """,
            limit, status, conversation_id,
        )
        return [dict(r) for r in rows]
