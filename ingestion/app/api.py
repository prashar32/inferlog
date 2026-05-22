"""Ingestion API.

Two jobs:
  * POST /v1/ingest  — accept event batches from the SDK, stamp them, and
    publish to the Redis stream. It does NOT touch Postgres; that's the
    worker's job. This keeps the write path fast and decoupled.
  * GET  /v1/metrics/* and /v1/logs — read side for the dashboards.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import IngestionDB
from .events import IngestBatch
from .stream import EventStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("ingestion.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = await IngestionDB.connect(settings.database_url)
    schema_file = Path(settings.schema_path)
    if schema_file.exists():
        await db.apply_schema(schema_file.read_text())

    stream = EventStream(
        settings.redis_url,
        stream_key=settings.stream_key,
        group=settings.consumer_group,
        dlq_key=settings.dlq_key,
    )
    await stream.ensure_group()

    app.state.db = db
    app.state.stream = stream
    log.info("ingestion API ready")
    try:
        yield
    finally:
        await stream.close()
        await db.close()


app = FastAPI(title="InferLog Ingestion", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_api_key(provided: str) -> None:
    if provided != settings.ingest_api_key:
        raise HTTPException(401, "invalid or missing x-api-key")


@app.post("/v1/ingest", status_code=202, tags=["ingest"])
async def ingest(batch: IngestBatch, x_api_key: str = Header(default="")):
    """Accept a batch of inference events. Returns immediately once they're
    on the stream — processing happens asynchronously in the worker."""
    _require_api_key(x_api_key)
    received_at = datetime.now(timezone.utc).isoformat()
    for event in batch.events:
        payload = event.model_dump(mode="json")
        payload["received_at"] = received_at
        await app.state.stream.publish(payload)
    return {"accepted": len(batch.events)}


@app.get("/v1/metrics/summary", tags=["metrics"])
async def metrics_summary(window: int = Query(60, ge=1, le=1440)):
    return await app.state.db.metrics_summary(window)


@app.get("/v1/metrics/timeseries", tags=["metrics"])
async def metrics_timeseries(
    window: int = Query(60, ge=1, le=1440),
    bucket: int = Query(60, ge=10, le=3600, description="bucket size in seconds"),
):
    return {
        "window_minutes": window,
        "bucket_seconds": bucket,
        "points": await app.state.db.metrics_timeseries(window, bucket),
    }


@app.get("/v1/metrics/errors", tags=["metrics"])
async def metrics_errors(window: int = Query(60, ge=1, le=1440)):
    return await app.state.db.metrics_errors(window)


@app.get("/v1/logs", tags=["logs"])
async def recent_logs(
    limit: int = Query(50, ge=1, le=200),
    status: str | None = Query(None),
    conversation_id: UUID | None = Query(None),
):
    return await app.state.db.recent_logs(limit, status, conversation_id)


@app.get("/healthz", tags=["meta"])
async def healthz():
    db_ok = await app.state.db.ping()
    redis_ok = await app.state.stream.ping()
    body = {"status": "ok" if (db_ok and redis_ok) else "degraded",
            "db": db_ok, "redis": redis_ok}
    if redis_ok:
        body["stream_depth"] = await app.state.stream.depth()
        body["pending"] = await app.state.stream.pending_count()
    return body
