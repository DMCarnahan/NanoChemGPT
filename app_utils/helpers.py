# app_utils/helpers.py
from __future__ import annotations
import os, re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterable, Optional

# ---------- small utils ----------
def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default

def _to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _to_str(x) -> str:
    if x is None: return ""
    return x if isinstance(x, str) else str(x)

# ---------- public helper API ----------
@dataclass
class Hit:
    i: int
    score: float
    text: str
    meta: Dict[str, Any]

    def asdict(self):
        return asdict(self)

def classify_intent(q: str) -> str:
    """Very cheap keyword classifier to unblock routing; extend later."""
    s = (q or "").lower()
    if any(k in s for k in ("how to", "steps", "procedure", "synthesize", "synthesis")):
        return "qa"
    if any(k in s for k in ("search", "find", "look up", "reference", "cite")):
        return "search"
    if any(k in s for k in ("summar", "abstract", "overview")):
        return "summary"
    return "qa"

def kb_search(q: str, top_k: int = 6) -> List[Hit]:
    """Minimal no-op search to avoid 500s if your real retriever isn't wired here."""
    return [] 

def kb_fetch(metas: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(metas)

def judge_sufficiency(
    hits: Iterable[Any],
    min_hits: Optional[int] = None,
    min_score: Optional[float] = None,
    min_chars: Optional[int] = None
) -> bool:
    """Return True if we have enough good hits. Robust to str/float/int inputs."""
    # allow per-call overrides, otherwise read env (and CAST!)
    min_hits  = int(min_hits) if min_hits is not None else env_int("JUDGE_MIN_HITS", 1)
    min_chars = int(min_chars) if min_chars is not None else env_int("JUDGE_MIN_CHARS", 48)
    min_score = float(min_score) if min_score is not None else env_float("JUDGE_MIN_SCORE", 0.0)

    good = 0
    for h in hits or []:
        if isinstance(h, dict):
            score = _to_float(h.get("score"), -1.0)
            text  = _to_str(h.get("text"))
        else:
            score = _to_float(getattr(h, "score", -1.0), -1.0)
            text  = _to_str(getattr(h, "text", ""))
        if len(text) >= min_chars and score >= min_score:
            good += 1
    return good >= min_hits

def judge_hits(hits, min_hits=1, min_score=0.0, min_chars=48):
    good = 0
    for h in hits or []:
        score = float(h.get("score", 0.0)) if isinstance(h, dict) else float(getattr(h, "score", 0.0))
        text  = (h.get("text", "") if isinstance(h, dict) else getattr(h, "text", "")) or ""
        if len(text) >= min_chars and score >= min_score:
            good += 1
    return good >= min_hits

# Trim answer safely
def _safe_text(s: str, max_chars: int = 8000) -> str:
    s = _to_str(s)
    if max_chars and max_chars > 0:
        return s[:max_chars]
    return s

# Extract citation indexes like [1], [2–3], etc.
_CIT_RX = re.compile(r"\[(\d+)(?:\s*[-–]\s*(\d+))?\]")
def _extract_used_ref_indexes(answer: str, default: Any = None) -> List[int]:
    s = _to_str(answer)
    found = []
    for m in _CIT_RX.finditer(s):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if a <= b:
            found.extend(range(a, b+1))
    # dedupe, preserve order
    seen, out = set(), []
    for i in found:
        if i not in seen:
            seen.add(i); out.append(i)
    return out if out else ([] if default is None else default)
