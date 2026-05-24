"""Async-safe tagging for inference events.

Lets the host attach context (conversation_id, user_id, tenant_id,
trace_id, ...) to every inference inside a scope, without touching the
call site. Implemented with contextvars so it propagates correctly across
`await` boundaries and is isolated per asyncio Task.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

_tags: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "inferlog_tags", default={}
)


@contextmanager
def context(**tags) -> Iterator[None]:
    """Add tags to every inference emitted in this scope.

        with inferlog.context(conversation_id=cid, user_id=uid):
            await client.chat.completions.create(...)
    """
    current = _tags.get()
    merged = {**current, **{k: v for k, v in tags.items() if v is not None}}
    token = _tags.set(merged)
    try:
        yield
    finally:
        _tags.reset(token)


def current_tags() -> dict:
    """Snapshot of the tags active for the current Task."""
    return dict(_tags.get())
