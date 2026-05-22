#!/usr/bin/env python3
"""Push synthetic inference logs to the ingestion API.

Useful for demos — it gives the dashboards a realistic shape without having
to sit and chat for ten minutes. The events go through the real pipeline
(ingest API -> Redis stream -> worker -> Postgres), so this also doubles as
a quick smoke test of the ingestion path.

Usage:
    python3 scripts/seed.py [count]      # default 80 events over the last hour

Standard library only, so it runs with a plain `python3` and no install.
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from uuid import uuid4

INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8081/v1/ingest")
API_KEY = os.getenv("INGEST_API_KEY", "local-dev-ingest-key")

MODELS = [
    ("openai", "gpt-4.1-mini"),
    ("openai", "gpt-4.1"),
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-sonnet-4-5"),
]
ERROR_TYPES = ["rate_limit", "timeout", "provider_error", "connection"]
# A couple of these contain PII on purpose — it should show up redacted.
SAMPLE_INPUTS = [
    "Summarise this quarter's revenue numbers",
    "email me at jordan@example.com about the rollout",
    "what is the capital of Australia",
    "call support on (415) 555-0199 if it breaks again",
    "write a haiku about distributed systems",
]


def make_event(started: datetime) -> dict:
    provider, model = random.choice(MODELS)
    streamed = random.random() < 0.8

    roll = random.random()
    status = "error" if roll < 0.08 else "cancelled" if roll < 0.13 else "success"

    latency_ms = max(120, int(random.gauss(1400, 600)))
    if status == "error":
        latency_ms = random.randint(150, 900)

    prompt = random.randint(20, 400)
    completion = 0 if status == "error" else random.randint(20, 600)

    return {
        "request_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "service": "chat-gateway",
        "provider": provider,
        "model": model,
        "status": status,
        "streamed": streamed,
        "started_at": started.isoformat(),
        "completed_at": (started + timedelta(milliseconds=latency_ms)).isoformat(),
        "latency_ms": latency_ms,
        "ttft_ms": random.randint(120, 500) if streamed and status != "error" else None,
        "prompt_tokens": prompt,
        "completion_tokens": completion or None,
        "total_tokens": (prompt + completion) if completion else None,
        "input_preview": random.choice(SAMPLE_INPUTS),
        "output_preview": None if status == "error" else "...(synthetic reply)...",
        "error_type": random.choice(ERROR_TYPES) if status == "error" else None,
        "error_message": "synthetic error for demo data" if status == "error" else None,
        "client_metadata": {"channel": "seed"},
        "sdk_version": "0.1.0",
        "schema_version": 1,
    }


def post(events: list[dict]) -> None:
    body = json.dumps({"events": events}).encode()
    request = urllib.request.Request(
        INGEST_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "x-api-key": API_KEY},
    )
    with urllib.request.urlopen(request, timeout=10):
        pass


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    now = datetime.now(timezone.utc)
    events = [
        make_event(now - timedelta(minutes=random.uniform(0, 58)))
        for _ in range(count)
    ]
    events.sort(key=lambda e: e["started_at"])

    for start in range(0, len(events), 25):
        post(events[start : start + 25])

    print(f"seeded {len(events)} synthetic events -> {INGEST_URL}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        sys.exit(
            f"could not reach the ingestion API at {INGEST_URL}: {exc}\n"
            f"is the stack running? try `make up` first."
        )
