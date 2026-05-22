"""Metadata the worker derives from a raw event before storing it.

None of this comes from the provider — it's computed during ingestion so
the dashboards can query it directly instead of recomputing on every read.
"""

from __future__ import annotations

# Approximate list price, USD per 1M tokens, as (input, output). These
# move over time — they live here precisely so there's one place to update.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-5": (3.00, 15.00),
    "mock-1": (0.0, 0.0),
}


def total_tokens(
    prompt: int | None, completion: int | None, reported: int | None
) -> int | None:
    """Trust the provider's total if given; otherwise reconstruct it."""
    if reported is not None:
        return reported
    if prompt is not None and completion is not None:
        return prompt + completion
    return None


def estimate_cost(
    model: str, prompt: int | None, completion: int | None
) -> float | None:
    rate = PRICING.get(model)
    if rate is None or prompt is None or completion is None:
        return None
    input_rate, output_rate = rate
    cost = prompt / 1_000_000 * input_rate + completion / 1_000_000 * output_rate
    return round(cost, 6)


def tokens_per_second(completion: int | None, latency_ms: int | None) -> float | None:
    if not completion or not latency_ms:
        return None
    return round(completion / (latency_ms / 1000), 2)
