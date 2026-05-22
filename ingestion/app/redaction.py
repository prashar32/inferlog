"""Regex-based PII redaction for log previews.

This is deliberately a small, well-understood set of patterns rather than
an ML model: it runs on every event, must be fast, and the failure mode we
care about (a leaked email/card number sitting in a log) is best caught by
something predictable. It is best-effort — see README for the tradeoff.
"""

from __future__ import annotations

import re

# Order matters: more specific patterns run first so a card number isn't
# half-eaten by the phone pattern, etc.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}\b")),
    ("PHONE", re.compile(
        r"\b(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"
    )),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def redact(text: str | None) -> tuple[str | None, int]:
    """Return (redacted_text, number_of_matches_replaced)."""
    if not text:
        return text, 0
    total = 0
    for label, pattern in _PATTERNS:
        text, count = pattern.subn(f"[REDACTED_{label}]", text)
        total += count
    return text, total
