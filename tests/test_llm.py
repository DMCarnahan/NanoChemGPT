from types import SimpleNamespace

import pytest

from app_utils.llm import (
    generate_text,
    load_model_settings,
    public_model_error,
    supports_reasoning,
)


class RecordingResponses:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def test_generate_text_uses_responses_api_contract():
    response = SimpleNamespace(
        output_text="  grounded answer  ",
        id="resp_123",
        model="gpt-6.1-sol-2026-09-01",
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 42}),
    )
    responses = RecordingResponses(response)
    client = SimpleNamespace(responses=responses)

    result = generate_text(
        client,
        model="gpt-6.1-sol",
        instructions="stable rules",
        input_text="request-specific evidence",
        reasoning_effort="high",
        max_output_tokens=2_000,
    )

    assert result.text == "grounded answer"
    assert result.response_id == "resp_123"
    assert result.model == "gpt-6.1-sol-2026-09-01"
    assert result.usage == {"input_tokens": 42}
    assert responses.requests == [
        {
            "model": "gpt-6.1-sol",
            "instructions": "stable rules",
            "input": "request-specific evidence",
            "max_output_tokens": 2_000,
            "store": False,
            "reasoning": {"effort": "high"},
        }
    ]


def test_non_reasoning_model_omits_reasoning_parameter():
    response = SimpleNamespace(output_text="answer", id=None, model=None, usage=None)
    responses = RecordingResponses(response)

    generate_text(
        SimpleNamespace(responses=responses),
        model="gpt-4o-mini",
        instructions="rules",
        input_text="question",
        reasoning_effort="high",
        max_output_tokens=512,
    )

    assert "reasoning" not in responses.requests[0]
    assert supports_reasoning("gpt-6.1-sol") is True
    assert supports_reasoning("gpt-4o-mini") is False


def test_model_settings_are_bounded(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "999999")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "99")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "invalid")

    settings = load_model_settings()

    assert settings.answer_model == "custom-model"
    assert settings.max_output_tokens == 32_000
    assert settings.timeout_seconds == 10.0
    assert settings.max_retries == 5
    assert settings.reasoning_effort == "high"


@pytest.mark.parametrize(
    ("exception", "code", "retryable"),
    [
        (TimeoutError(), "model_timeout", True),
        (type("AuthenticationError", (Exception,), {})(), "invalid_api_key", False),
        (
            type("APIConnectionError", (Exception,), {})(),
            "model_connection_error",
            True,
        ),
    ],
)
def test_public_model_error_mapping(exception, code, retryable):
    result = public_model_error(exception)
    assert result.code == code
    assert result.retryable is retryable
