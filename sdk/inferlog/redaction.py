"""PII redaction — runs inside the host application, before anything crosses
the wire.

This is a deliberate inversion of where the take-home put it. The whole
point of redaction is that sensitive bytes never leave the boundary they
were created in. If we redacted on the ingestion side we'd already have
raw PII in transit to our infrastructure — which defeats the purpose for
any customer with a real compliance posture (HIPAA, GDPR, SOC 2).

The default is a small, well-understood regex pass. Customers can:
  * add their own patterns (e.g. an internal customer-ID format),
  * disable it entirely (with a loud warning in init),
  * or plug in their own implementation — Presidio, spaCy NER, an LLM.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Pattern

# Order matters: more specific patterns run first so a card number isn't
# half-eaten by the phone pattern, etc.
_DEFAULT_PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}\b")),
    ("PHONE", re.compile(
        r"\b(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"
    )),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


class Redactor:
    """Pluggable redactor.

    The default behaviour is the regex pass above. Replace `custom` with
    your own callable for ML-grade redaction (Presidio, NER, LLM judge).
    `extra_patterns` lets you keep the defaults and add company-specific
    ones like internal IDs.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        extra_patterns: Iterable[tuple[str, str | Pattern[str]]] | None = None,
        custom: Callable[[str], tuple[str, int]] | None = None,
    ):
        self.enabled = enabled
        self._custom = custom
        self._patterns: list[tuple[str, Pattern[str]]] = list(_DEFAULT_PATTERNS)
        if extra_patterns:
            for label, pat in extra_patterns:
                compiled = re.compile(pat) if isinstance(pat, str) else pat
                self._patterns.append((label, compiled))

    def redact(self, text: str | None) -> tuple[str | None, int]:
        """Return (redacted_text, number_of_matches_replaced)."""
        if not self.enabled or not text:
            return text, 0
        if self._custom is not None:
            # Trust the customer's redactor; we don't second-guess it.
            return self._custom(text)
        total = 0
        for label, pattern in self._patterns:
            text, count = pattern.subn(f"[REDACTED_{label}]", text)
            total += count
        return text, total
