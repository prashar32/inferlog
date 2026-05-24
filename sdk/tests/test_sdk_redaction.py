"""The SDK must redact PII BEFORE the event leaves the host process.

These tests assert the contract: whatever lands in the sink has previews
that contain no raw PII and a `pii_redaction_count` describing how many
substitutions were made.
"""

import asyncio

from inferlog import (
    HttpSink,
    LogDispatcher,
    LoggedLLMClient,
    MemorySink,
    Redactor,
)
from inferlog.providers import ChatMessage, MockProvider


async def _drain(dispatcher: LogDispatcher) -> None:
    await asyncio.sleep(0.05)
    await dispatcher.aclose()


async def _run_one_chat(prompt: str, redactor: Redactor | None = None):
    sink = MemorySink()
    dispatcher = LogDispatcher(sink, flush_interval=0.05)
    dispatcher.start()
    client = LoggedLLMClient(
        service="t",
        dispatcher=dispatcher,
        providers={"mock": MockProvider(token_delay=0)},
        redactor=redactor or Redactor(),
    )
    await client.complete(
        provider="mock", model="mock-1",
        messages=[ChatMessage("user", prompt)],
    )
    await _drain(dispatcher)
    assert len(sink.events) == 1
    return sink.events[0]


async def test_email_is_redacted_before_reaching_sink():
    event = await _run_one_chat("contact me at jane.doe@example.com tomorrow")
    assert "jane.doe@example.com" not in event["input_preview"]
    assert "[REDACTED_EMAIL]" in event["input_preview"]
    # mock echoes the user message, so the output preview also contained the
    # email — both should be redacted, count covers both.
    assert event["pii_redaction_count"] >= 1


async def test_multiple_pii_kinds_in_one_call():
    event = await _run_one_chat(
        "card 4111 1111 1111 1111 ssn 123-45-6789 host 10.0.0.5"
    )
    assert "4111 1111 1111 1111" not in event["input_preview"]
    assert "123-45-6789" not in event["input_preview"]
    assert "10.0.0.5" not in event["input_preview"]
    assert event["pii_redaction_count"] >= 3


async def test_redactor_can_be_disabled():
    event = await _run_one_chat("call me on jane@x.com", redactor=Redactor(enabled=False))
    assert "jane@x.com" in event["input_preview"]
    assert event["pii_redaction_count"] == 0


async def test_extra_patterns_apply():
    redactor = Redactor(extra_patterns=[("INTERNAL_ID", r"INT-\d{6}")])
    event = await _run_one_chat("the ticket is INT-987654, please look", redactor=redactor)
    assert "INT-987654" not in event["input_preview"]
    assert "[REDACTED_INTERNAL_ID]" in event["input_preview"]


async def test_custom_redactor_replaces_default():
    def all_caps_redactor(text: str):
        return f"[REDACTED {len(text)}]", 1
    redactor = Redactor(custom=all_caps_redactor)
    event = await _run_one_chat("any text at all", redactor=redactor)
    assert event["input_preview"].startswith("[REDACTED ")


def test_http_sink_payload_already_has_redacted_previews(event_factory):
    """Belt-and-braces — the on-the-wire payload is what the redactor produced."""
    # Build an event with a redacted preview directly.
    redactor = Redactor()
    redacted, count = redactor.redact("ping me at zoe@y.com please")
    event = event_factory(input_preview=redacted, pii_redaction_count=count)
    payload = event.to_payload()
    assert "zoe@y.com" not in (payload["input_preview"] or "")
    assert payload["pii_redaction_count"] == 1
