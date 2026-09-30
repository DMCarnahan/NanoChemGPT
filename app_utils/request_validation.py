"""Validation helpers for public API request controls."""

from __future__ import annotations

import math
import re
from typing import Any

ATTACHMENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
VALID_MODES = {"protocol", "reasoning"}
MODE_ALIASES = {"robot": "protocol", "analysis": "reasoning", "reason": "reasoning"}


class RequestValidationError(ValueError):
    pass


def bounded_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def bounded_float(
    value: Any, default: float, *, minimum: float, maximum: float
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(minimum, min(parsed, maximum))


def normalize_mode(value: Any, *, default: str = "protocol") -> str:
    mode = str(value or default).strip().lower()
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in VALID_MODES:
        raise RequestValidationError(
            f"Unsupported mode '{mode}'. Use 'protocol' or 'reasoning'."
        )
    return mode


def normalize_attachment_ids(value: Any, *, maximum: int = 5) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise RequestValidationError("attachments must be a list of attachment IDs")

    normalized: list[str] = []
    for raw in values:
        attachment_id = str(raw or "").strip()
        if not attachment_id:
            continue
        if not ATTACHMENT_ID_RE.fullmatch(attachment_id):
            raise RequestValidationError("Invalid attachment ID")
        if attachment_id not in normalized:
            normalized.append(attachment_id)
        if len(normalized) >= maximum:
            break
    return normalized


def truncate_context(value: str, *, maximum_chars: int) -> str:
    """Bound source context while making truncation explicit to the model."""

    text = (value or "").strip()
    if maximum_chars <= 0:
        return ""
    if len(text) <= maximum_chars:
        return text
    marker = "\n\n[Context truncated by server at configured limit.]"
    if len(marker) >= maximum_chars:
        return marker[-maximum_chars:]
    return text[: max(0, maximum_chars - len(marker))].rstrip() + marker
