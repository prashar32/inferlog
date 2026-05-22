"""Gateway service — the chatbot backend.

Owns conversations + messages, talks to LLM providers through the inferlog
SDK, and streams answers to the UI. Inference logs leave via the SDK; this
service never writes to `inference_logs` itself.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import chat, conversations
from .config import settings
from .db import Database
from .llm import LLMRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = await Database.connect(settings.database_url)
    schema_file = Path(settings.schema_path)
    if schema_file.exists():
        # Idempotent. Also runs in the Postgres init script; harmless twice.
        await db.apply_schema(schema_file.read_text())
        log.info("schema ensured from %s", schema_file)
    else:
        log.warning("schema file %s missing — assuming DB already migrated", schema_file)

    llm = LLMRuntime.build(settings)
    llm.start()

    app.state.db = db
    app.state.llm = llm
    log.info("gateway ready")
    try:
        yield
    finally:
        await llm.aclose()
        await db.close()


app = FastAPI(title="InferLog Gateway", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(conversations.router)
app.include_router(chat.router)


@app.get("/healthz", tags=["meta"])
async def healthz():
    db_ok = await app.state.db.ping()
    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "logging": app.state.llm.logging_stats(),
    }


@app.get("/v1/models", tags=["meta"])
async def list_models():
    """Models the UI may offer. First entry is the suggested default."""
    llm: LLMRuntime = app.state.llm
    return [
        {"provider": m.provider, "model": m.model, "label": m.label}
        for m in llm.available_models
    ]
