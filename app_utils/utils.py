"""Small utility helpers used by app.py (sanitizers, converters, markers)."""
from __future__ import annotations

import re
from typing import Any


def s(x: Any) -> str:
    try:
        from app_utils.helpers import _safe_text as _h_safe_text
        return _h_safe_text(x)
    except Exception:
        return str(x).strip() if x is not None else ""


def safe_id(x: Any):
    # Treat falsy values (None, empty string) as None to match app expectations
    if not x:
        return None
    try:
        from bson import ObjectId
        return ObjectId(x)
    except Exception:
        return None


def stringify_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [stringify_keys(x) for x in obj]
    return obj


def doc(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    for k, v in out.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out


def extract_used_markers(*texts: str) -> dict:
    _CIT_BRACKET_RX = re.compile(r"\[(?P<num>\d{1,4})\]")
    _CIT_FULL_RX = re.compile(r"【(?P<num>\d{1,4})】")
    _CIT_FOOT_RX = re.compile(r"\[\^(?P<num>\d{1,4})\]")
    TAGS = ("CTX", "PARSED", "DB", "GEN")
    seen = set()
    tag_counts = {t: 0 for t in TAGS}
    for t in texts:
        if not t: continue
        for rx in (_CIT_BRACKET_RX, _CIT_FULL_RX, _CIT_FOOT_RX):
            for m in rx.finditer(t):
                try: seen.add(int(m.group("num")))
                except Exception: pass
        for tag in TAGS:
            tag_counts[tag] += len(re.findall(rf"\[{tag}\]", t))
    return {"refs": sorted(seen), "tags": tag_counts, "has_ctx": any(tag_counts[k] > 0 for k in ("CTX", "PARSED", "DB"))}


def wants_verbatim(q: str) -> bool:
    q = (q or "").lower()
    keys = ("repeat", "verbatim", "quote", "transcribe", "as written", "exact text")
    return any(k in q for k in keys)


def clean_verbatim_block(s: str) -> str:
    s = re.sub(r"^\[A\d+\.\d+\]\s*attachment:[^\n]+\n", "", s, flags=re.M)
    s = s.replace("", "·")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()
