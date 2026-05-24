"""Global SDK runtime — what `inferlog.init()` constructs.

There's exactly one of these per process. It owns the dispatcher, the
redactor, and the few pieces of static config the auto-instrumentation
and the explicit client both need to find.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .dispatcher import HttpSink, LogDispatcher, NullSink, Sink
from .redaction import Redactor
from .sampling import KeepAll, Sampler

log = logging.getLogger("inferlog.runtime")


@dataclass
class Runtime:
    service: str
    dispatcher: LogDispatcher
    redactor: Redactor
    sampler: Sampler
    enabled: bool = True
    sdk_options: dict = field(default_factory=dict)

    def shutdown(self) -> None:
        # Convenience for sync callers; async path uses dispatcher.aclose().
        self.enabled = False


_runtime: Runtime | None = None


def get_runtime() -> Runtime | None:
    return _runtime


def set_runtime(rt: Runtime | None) -> None:
    global _runtime
    _runtime = rt


def is_initialized() -> bool:
    return _runtime is not None


def build_default_runtime(
    *,
    service: str,
    endpoint: str | None,
    api_key: str | None,
    enabled: bool = True,
    redactor: Redactor | None = None,
    sink: Sink | None = None,
    sampler: Sampler | None = None,
    auth_scheme: str = "x-api-key",
    dispatcher_options: dict | None = None,
) -> Runtime:
    """Construct a Runtime with sensible defaults.

    `endpoint=None` (or `enabled=False`) wires the dispatcher to a
    NullSink so init() is effectively a no-op (useful for tests and CI).
    """
    if sink is None:
        if endpoint and enabled:
            sink = HttpSink(endpoint, api_key=api_key, auth_scheme=auth_scheme)
        else:
            sink = NullSink()
    dispatcher = LogDispatcher(sink, **(dispatcher_options or {}))
    return Runtime(
        service=service,
        dispatcher=dispatcher,
        redactor=redactor or Redactor(),
        sampler=sampler or KeepAll(),
        enabled=enabled,
    )
