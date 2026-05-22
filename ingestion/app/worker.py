"""Ingestion worker — consumes the Redis stream and writes `inference_logs`.

Delivery model: at-least-once. A message is acked only after its row is
committed (or parked in the DLQ). The DB upsert is idempotent on
request_id, so a redelivery after a crash is harmless. On startup the
worker also reclaims messages a previous, dead worker left pending.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
from pathlib import Path

from pydantic import ValidationError

from .config import settings
from .db import IngestionDB
from .events import IngestEvent
from .processing import process_event
from .stream import EventStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("ingestion.worker")


async def _handle(stream: EventStream, db: IngestionDB, messages) -> None:
    for msg_id, raw in messages:
        try:
            event = IngestEvent.model_validate(raw)
        except ValidationError as exc:
            # Bad shape — it will never succeed, so don't retry it forever.
            await stream.to_dlq(raw, f"validation error: {exc}")
            await stream.ack(msg_id)
            continue

        try:
            record = process_event(event)
            inserted = await db.upsert_log(record)
            await stream.ack(msg_id)
            if not inserted:
                log.info("event %s already stored — idempotent skip", event.request_id)
        except Exception as exc:  # noqa: BLE001
            # Transient infra errors would ideally be retried; for this
            # system we DLQ-and-continue so one bad event can't wedge the
            # stream. The DLQ is the audit trail for that decision.
            log.exception("failed to store event %s", event.request_id)
            await stream.to_dlq(raw, f"processing error: {exc}")
            await stream.ack(msg_id)


async def run() -> None:
    consumer = f"{socket.gethostname()}-{os.getpid()}"

    db = await IngestionDB.connect(settings.database_url)
    schema_file = Path(settings.schema_path)
    if schema_file.exists():
        # The worker may win the startup race against the gateway.
        await db.apply_schema(schema_file.read_text())

    stream = EventStream(
        settings.redis_url,
        stream_key=settings.stream_key,
        group=settings.consumer_group,
        dlq_key=settings.dlq_key,
    )
    await stream.ensure_group()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    log.info("worker '%s' started, consuming '%s'", consumer, settings.stream_key)

    # Pick up anything a previous worker took but never acked.
    reclaimed = await stream.claim_stale(
        consumer, settings.claim_idle_ms, settings.batch_size
    )
    if reclaimed:
        log.info("reclaimed %d pending message(s) on startup", len(reclaimed))
        await _handle(stream, db, reclaimed)

    while not stop.is_set():
        messages = await stream.read(consumer, settings.batch_size, settings.block_ms)
        if messages:
            await _handle(stream, db, messages)
        else:
            # Idle — also a good moment to sweep for stale pending messages.
            reclaimed = await stream.claim_stale(
                consumer, settings.claim_idle_ms, settings.batch_size
            )
            if reclaimed:
                await _handle(stream, db, reclaimed)

    log.info("worker '%s' shutting down", consumer)
    await stream.close()
    await db.close()


if __name__ == "__main__":
    asyncio.run(run())
