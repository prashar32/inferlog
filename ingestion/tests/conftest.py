"""Ingestion test fixtures.

Pure tests (redaction, enrichment, processing) need nothing. The API and
worker tests need a real Postgres and Redis — `make test` provides both via
docker compose. Each test gets its own Redis stream key so they don't see
each other's events.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest_asyncio
from httpx import ASGITransport

from app import api as api_module
from app.config import settings
from app.db import IngestionDB
from app.stream import EventStream

def _load_schema() -> str:
    # In the container SCHEMA_PATH is set; from the repo, fall back to the
    # path relative to this file (ingestion/tests/ -> repo root -> db/).
    env_path = os.getenv("SCHEMA_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path).read_text()
    return (Path(__file__).resolve().parents[2] / "db" / "schema.sql").read_text()


SCHEMA_SQL = _load_schema()
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://inferlog:inferlog@postgres:5432/inferlog"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
API_KEY = "test-ingest-key"


def event_payload(**overrides) -> dict:
    """A JSON-shaped event, as it would arrive from the SDK."""
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "request_id": str(uuid.uuid4()),
        "service": "chat-gateway",
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "status": "success",
        "streamed": True,
        "started_at": now,
        "completed_at": now,
        "latency_ms": 900,
        "ttft_ms": 200,
        "prompt_tokens": 40,
        "completion_tokens": 80,
        "total_tokens": 120,
        "input_preview": "hello",
        "output_preview": "hi there",
        "error_type": None,
        "error_message": None,
        "client_metadata": {"channel": "web"},
        "sdk_version": "0.1.0",
        "schema_version": 1,
    }
    payload.update(overrides)
    return payload


@pytest_asyncio.fixture
async def db():
    database = await IngestionDB.connect(DATABASE_URL)
    await database.apply_schema(SCHEMA_SQL)
    async with database._pool.acquire() as conn:
        await conn.execute("TRUNCATE inference_logs")
    yield database
    await database.close()


@pytest_asyncio.fixture
async def stream():
    suffix = uuid.uuid4().hex[:8]
    bus = EventStream(
        REDIS_URL,
        stream_key=f"test:events:{suffix}",
        group="workers",
        dlq_key=f"test:dlq:{suffix}",
    )
    await bus.ensure_group()
    yield bus
    await bus._redis.delete(bus._stream, bus._dlq)
    await bus.close()


@pytest_asyncio.fixture
async def client(db, stream):
    settings.ingest_api_key = API_KEY
    api_module.app.state.db = db
    api_module.app.state.stream = stream
    transport = ASGITransport(app=api_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
