import os
from dataclasses import dataclass, field


def _csv(name: str, default: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, default).split(",") if v.strip()]


@dataclass
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://inferlog:inferlog@postgres:5432/inferlog"
    )
    # Where the SDK ships inference logs. The gateway never writes logs to
    # the DB directly — it goes through the ingestion service.
    ingest_url: str = os.getenv("INGEST_URL", "http://ingestion-api:8080/v1/ingest")
    ingest_api_key: str = os.getenv("INGEST_API_KEY", "local-dev-ingest-key")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "").strip()

    # How many recent messages to replay to the model. This is the "short
    # conversational context" — a plain sliding window, not summarisation.
    context_window: int = int(os.getenv("CONTEXT_WINDOW_MESSAGES", "12"))
    default_system_prompt: str = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are InferLog's demo assistant. Be helpful and concise.",
    )

    schema_path: str = os.getenv("SCHEMA_PATH", "/app/db/schema.sql")
    cors_origins: list[str] = field(default_factory=lambda: _csv("CORS_ORIGINS", "*"))
    port: int = int(os.getenv("PORT", "8080"))


settings = Settings()
