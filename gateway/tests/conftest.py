"""Gateway test fixtures.

These tests need a real Postgres (the queries use Postgres-specific SQL).
`make test` brings one up via docker compose; DATABASE_URL points at it.
The LLM side is wired to the offline mock provider and an in-memory log
sink, so tests need no API keys and make no network calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from inferlog import LoggedLLMClient, LogDispatcher, MemorySink
from inferlog.providers import MockProvider

from app.db import Database
from app.llm import LLMRuntime
from app.main import app

def _load_schema() -> str:
    # In the container SCHEMA_PATH is set; from the repo, fall back to the
    # path relative to this file (gateway/tests/ -> repo root -> db/).
    env_path = os.getenv("SCHEMA_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path).read_text()
    return (Path(__file__).resolve().parents[2] / "db" / "schema.sql").read_text()


SCHEMA_SQL = _load_schema()
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://inferlog:inferlog@postgres:5432/inferlog"
)


@pytest_asyncio.fixture
async def db():
    database = await Database.connect(DATABASE_URL)
    await database.apply_schema(SCHEMA_SQL)
    async with database._pool.acquire() as conn:
        await conn.execute("TRUNCATE conversations, messages CASCADE")
    yield database
    await database.close()


@pytest_asyncio.fixture
async def sink():
    return MemorySink()


@pytest_asyncio.fixture
async def client(db, sink):
    # Small token delay so streaming has several steps — enough for the
    # cancellation test to interrupt mid-stream.
    dispatcher = LogDispatcher(sink, flush_interval=0.05)
    dispatcher.start()
    providers = {"mock": MockProvider(token_delay=0.02)}
    llm = LLMRuntime(
        LoggedLLMClient(service="test-gateway", dispatcher=dispatcher, providers=providers),
        providers,
    )
    app.state.db = db
    app.state.llm = llm

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await dispatcher.aclose()


async def create_conversation(client: httpx.AsyncClient, model: str = "mock-1") -> str:
    resp = await client.post("/v1/conversations", json={"model": model})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def parse_sse(text: str) -> list[tuple[str, str]]:
    """Parse an SSE response body into (event, data) pairs."""
    events: list[tuple[str, str]] = []
    event = "message"
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            events.append((event, line[len("data:"):].strip()))
            event = "message"
    return events
