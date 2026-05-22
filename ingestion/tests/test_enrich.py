from app.enrich import estimate_cost, tokens_per_second, total_tokens


def test_total_tokens_prefers_reported_value():
    assert total_tokens(10, 20, 99) == 99


def test_total_tokens_reconstructs_when_missing():
    assert total_tokens(10, 20, None) == 30


def test_total_tokens_none_when_unknowable():
    assert total_tokens(None, 20, None) is None


def test_estimate_cost_for_known_model():
    # 1M input + 1M output at gpt-4.1-mini's (0.40, 1.60) rate.
    assert estimate_cost("gpt-4.1-mini", 1_000_000, 1_000_000) == 2.0


def test_estimate_cost_unknown_model_is_none():
    assert estimate_cost("some-future-model", 100, 100) is None


def test_tokens_per_second():
    assert tokens_per_second(100, 1000) == 100.0
    assert tokens_per_second(None, 1000) is None
    assert tokens_per_second(100, 0) is None
