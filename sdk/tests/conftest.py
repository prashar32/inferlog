"""Shared test helpers for the inferlog SDK."""

from __future__ import annotations

from datetime import timedelta

import pytest

from inferlog.events import InferenceEvent, utcnow


def make_event(**overrides) -> InferenceEvent:
    """A valid InferenceEvent with sensible defaults; override what matters."""
    started = utcnow()
    defaults = dict(
        request_id="00000000-0000-0000-0000-000000000001",
        service="test",
        provider="mock",
        model="mock-1",
        status="success",
        streamed=False,
        started_at=started,
        completed_at=started + timedelta(milliseconds=120),
        latency_ms=120,
    )
    defaults.update(overrides)
    return InferenceEvent(**defaults)


@pytest.fixture
def event_factory():
    return make_event
