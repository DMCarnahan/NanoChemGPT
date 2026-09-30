"""OpenAI client configuration and text-generation helpers.

The web app used to construct Chat Completions requests inline in ``app.py``.
Keeping the API boundary here makes model selection explicit, keeps request
timeouts consistent with Railway, and gives callers one safe error vocabulary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
_VALID_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _effort(name: str, default: str) -> str:
    value = (os.getenv(name) or default).strip().lower()
    return value if value in _VALID_REASONING_EFFORTS else default


@dataclass(frozen=True)
class ModelSettings:
    """Runtime model settings, loaded from environment variables."""

    answer_model: str
    citation_model: str
    reasoning_effort: str
    citation_reasoning_effort: str
    max_output_tokens: int
    citation_max_output_tokens: int
    timeout_seconds: float
    max_retries: int


def load_model_settings() -> ModelSettings:
    """Return validated settings without caching environment state."""

    return ModelSettings(
        answer_model=(os.getenv("OPENAI_MODEL") or "gpt-6.1-sol").strip(),
        citation_model=(os.getenv("OPENAI_CITATION_MODEL") or "gpt-6-luna").strip(),
        reasoning_effort=_effort("OPENAI_REASONING_EFFORT", "high"),
        citation_reasoning_effort=_effort("OPENAI_CITATION_REASONING_EFFORT", "none"),
        max_output_tokens=_env_int(
            "OPENAI_MAX_OUTPUT_TOKENS", 10_000, minimum=512, maximum=32_000
        ),
        citation_max_output_tokens=_env_int(
            "OPENAI_CITATION_MAX_OUTPUT_TOKENS",
            4_000,
            minimum=256,
            maximum=8_000,
        ),
        timeout_seconds=_env_float(
            "OPENAI_TIMEOUT_SECONDS", 120.0, minimum=10.0, maximum=300.0
        ),
        max_retries=_env_int("OPENAI_MAX_RETRIES", 2, minimum=0, maximum=5),
    )


def create_openai_client() -> OpenAI | None:
    """Create the process-wide OpenAI client, or ``None`` when unconfigured."""

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    settings = load_model_settings()
    timeout = httpx.Timeout(
        settings.timeout_seconds,
        connect=min(15.0, settings.timeout_seconds),
        write=min(30.0, settings.timeout_seconds),
        pool=min(15.0, settings.timeout_seconds),
    )
    http_client = httpx.Client(
        timeout=timeout,
        trust_env=_env_bool("OPENAI_TRUST_ENV", False),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "http_client": http_client,
        "max_retries": settings.max_retries,
    }
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def supports_reasoning(model: str) -> bool:
    """Return whether a model name belongs to a reasoning-model family."""

    name = (model or "").lower()
    return name.startswith(_REASONING_PREFIXES)


@dataclass(frozen=True)
class GeneratedText:
    text: str
    model: str
    response_id: str | None
    usage: dict[str, Any]


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Small compatibility path for hand-written test doubles and SDK variants.
    pieces: list[str] = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            value = getattr(part, "text", None)
            if isinstance(value, str):
                pieces.append(value)
    return "\n".join(pieces).strip()


def generate_text(
    client: Any,
    *,
    model: str,
    instructions: str,
    input_text: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> GeneratedText:
    """Generate text through the Responses API.

    Source material belongs in ``input_text`` while stable application rules
    belong in ``instructions``. This preserves instruction priority and makes
    the stable prefix eligible for prompt caching.
    """

    request: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if supports_reasoning(model):
        request["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request)
    text = _response_text(response)
    if not text:
        raise RuntimeError("The language model returned an empty response")

    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        usage = {}
    return GeneratedText(
        text=text,
        model=str(getattr(response, "model", None) or model),
        response_id=getattr(response, "id", None),
        usage=usage,
    )


@dataclass(frozen=True)
class PublicModelError:
    code: str
    message: str
    retryable: bool
    request_id: str | None = None


def _upstream_error_code(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if code:
        return str(code).lower()
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(error, dict) and error.get("code"):
            return str(error["code"]).lower()
    return ""


def public_model_error(exc: Exception) -> PublicModelError:
    """Map SDK/network exceptions to useful messages without leaking details."""

    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    upstream_code = _upstream_error_code(exc)
    request_id = getattr(exc, "request_id", None)

    if status == 401 or name == "AuthenticationError":
        return PublicModelError(
            "invalid_api_key",
            "OpenAI rejected the API key configured for this deployment.",
            False,
            request_id,
        )
    if status == 403 or name == "PermissionDeniedError":
        return PublicModelError(
            "model_permission_denied",
            "This OpenAI project is not allowed to use the configured model.",
            False,
            request_id,
        )
    if status == 429 or name == "RateLimitError":
        if upstream_code in {
            "insufficient_quota",
            "billing_hard_limit_reached",
            "billing_not_active",
        }:
            return PublicModelError(
                "quota_exceeded",
                "The OpenAI project has no available quota or billing capacity.",
                False,
                request_id,
            )
        return PublicModelError(
            "rate_limited",
            "OpenAI is rate-limiting requests; try again shortly.",
            True,
            request_id,
        )
    if status == 404 or upstream_code in {"model_not_found", "unknown_model"}:
        return PublicModelError(
            "model_unavailable",
            "The configured OpenAI model is unavailable to this project.",
            False,
            request_id,
        )
    if name in {"APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout"}:
        return PublicModelError(
            "model_timeout",
            "The language model took too long to respond; try again.",
            True,
            request_id,
        )
    if name in {"APIConnectionError", "ConnectError", "NetworkError"}:
        return PublicModelError(
            "model_connection_error",
            "The server could not reach OpenAI; try again shortly.",
            True,
            request_id,
        )
    if isinstance(status, int) and status >= 500:
        return PublicModelError(
            "model_service_error",
            "OpenAI is temporarily unavailable; try again shortly.",
            True,
            request_id,
        )
    return PublicModelError(
        "model_request_failed",
        "Language model request failed.",
        False,
        request_id,
    )
