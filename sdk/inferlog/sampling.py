"""Sampling — decide which events to ship at high volume.

Every production observability SDK has this. Two reasons:

  * Cost — at 10k requests/sec, storing every event is expensive and
    rarely useful. p99 latency over a 1% sample is identical.
  * Privacy — some customers want random sampling so any given inference
    has only a probabilistic chance of being persisted.

Default is `KeepAll`. Wrap any sampler in `AlwaysKeepErrors` to ensure
errors are never sampled out — which is almost always what you want.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable

from .events import InferenceEvent


class Sampler(ABC):
    @abstractmethod
    def should_sample(self, event: InferenceEvent) -> bool: ...


class KeepAll(Sampler):
    def should_sample(self, event: InferenceEvent) -> bool:  # noqa: ARG002
        return True


class Probability(Sampler):
    """Independent Bernoulli sampler with rate in [0, 1]."""

    def __init__(self, rate: float):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate!r}")
        self._rate = rate

    def should_sample(self, event: InferenceEvent) -> bool:  # noqa: ARG002
        return self._rate >= 1.0 or random.random() < self._rate


class AlwaysKeepErrors(Sampler):
    """Wraps another sampler; errors always pass, successes flow through.

    Almost always what you want — sampling away the rare failure is the
    worst possible outcome of sampling.
    """

    def __init__(self, inner: Sampler):
        self._inner = inner

    def should_sample(self, event: InferenceEvent) -> bool:
        if event.status in ("error", "cancelled"):
            return True
        return self._inner.should_sample(event)


class CustomSampler(Sampler):
    """Adapter for a plain callable — `lambda event: bool`."""

    def __init__(self, fn: Callable[[InferenceEvent], bool]):
        self._fn = fn

    def should_sample(self, event: InferenceEvent) -> bool:
        try:
            return bool(self._fn(event))
        except Exception:  # noqa: BLE001 — sampler bugs must not break logging
            return True
