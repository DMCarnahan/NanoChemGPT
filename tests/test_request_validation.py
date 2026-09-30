import pytest

from app_utils.request_validation import (
    RequestValidationError,
    bounded_float,
    bounded_int,
    normalize_attachment_ids,
    normalize_mode,
    truncate_context,
)


def test_numeric_controls_are_bounded_and_finite():
    assert bounded_int("200", 5, minimum=1, maximum=20) == 20
    assert bounded_int("bad", 5, minimum=1, maximum=20) == 5
    assert bounded_float("nan", 0.6, minimum=0.0, maximum=1.0) == 0.6
    assert bounded_float("-3", 0.6, minimum=0.0, maximum=1.0) == 0.0


def test_mode_aliases_and_invalid_mode():
    assert normalize_mode("robot") == "protocol"
    assert normalize_mode("analysis") == "reasoning"
    with pytest.raises(RequestValidationError):
        normalize_mode("surprise")


def test_attachment_ids_are_safe_deduplicated_and_limited():
    assert normalize_attachment_ids("abc,abc,def") == ["abc", "def"]
    assert normalize_attachment_ids(["a", "b", "c"], maximum=2) == ["a", "b"]
    with pytest.raises(RequestValidationError):
        normalize_attachment_ids("../../secrets")


def test_context_truncation_is_explicit():
    result = truncate_context("x" * 100, maximum_chars=50)
    assert len(result) == 50
    assert result.endswith("[Context truncated by server at configured limit.]")
