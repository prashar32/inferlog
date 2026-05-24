import os
from dataclasses import dataclass, field


def _csv(name: str, default: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, default).split(",") if v.strip()]


@dataclass
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://inferlog:inferlog@postgres:5432/inferlog"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # Shared secret the SDK must present on POST /v1/ingest.
    ingest_api_key: str = os.getenv("INGEST_API_KEY", "local-dev-ingest-key")

    # Redis Streams topology.
    stream_key: str = os.getenv("STREAM_KEY", "inferlog:events")
    consumer_group: str = os.getenv("CONSUMER_GROUP", "workers")
    dlq_key: str = os.getenv("DLQ_KEY", "inferlog:events:dlq")

    # Worker tuning.
    batch_size: int = int(os.getenv("WORKER_BATCH_SIZE", "50"))
    block_ms: int = int(os.getenv("WORKER_BLOCK_MS", "5000"))
    # A message pending longer than this is assumed orphaned by a dead
    # worker and reclaimed via XAUTOCLAIM.
    claim_idle_ms: int = int(os.getenv("WORKER_CLAIM_IDLE_MS", "30000"))

    schema_path: str = os.getenv("SCHEMA_PATH", "/app/db/schema.sql")
    # Default to the local dashboard host. Override with CORS_ORIGINS=*
    # for development against a separate origin (e.g. the vite dev server).
    # A `*` default would let any site read all customer logs via the
    # browser — dashboard endpoints are auth-gated, but defence-in-depth.
    cors_origins: list[str] = field(
        default_factory=lambda: _csv(
            "CORS_ORIGINS", "http://localhost:8088,http://localhost:5173"
        )
    )
    port: int = int(os.getenv("PORT", "8080"))


settings = Settings()
