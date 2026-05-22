from app.redaction import redact


def test_redacts_email():
    out, count = redact("ping me at jane.doe@example.com when ready")
    assert "jane.doe@example.com" not in out
    assert "[REDACTED_EMAIL]" in out
    assert count == 1


def test_redacts_several_kinds_at_once():
    text = "card 4111 1111 1111 1111, ssn 123-45-6789, host 10.0.0.5"
    out, count = redact(text)
    assert "[REDACTED_CARD]" in out
    assert "[REDACTED_SSN]" in out
    assert "[REDACTED_IP]" in out
    assert count == 3


def test_redacts_phone_number():
    out, count = redact("call (415) 555-2671 after noon")
    assert "[REDACTED_PHONE]" in out
    assert count == 1


def test_redacts_api_key():
    out, count = redact("key=sk-abcdEFGH1234abcdEFGH live")
    assert "[REDACTED_API_KEY]" in out
    assert count == 1


def test_clean_text_is_untouched():
    text = "a perfectly ordinary sentence about the weather"
    out, count = redact(text)
    assert out == text
    assert count == 0


def test_handles_empty_and_none():
    assert redact(None) == (None, 0)
    assert redact("") == ("", 0)
