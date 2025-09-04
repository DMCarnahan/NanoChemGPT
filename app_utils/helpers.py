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
_CITE_RX = re.compile(r'\[(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\]')

def extract_used_ref_indexes(*texts):
    used = set()
    for t in texts:
        if not t: continue
        for m in _CITE_RX.finditer(t):
            chunk = m.group(1)
            for part in re.split(r'\s*,\s*', chunk):
                if re.search(r'[-–]', part):
                    a, b = re.split(r'[-–]', part)
                    for i in range(int(a), int(b) + 1):
                        used.add(i)
                else:
                    used.add(int(part))
    return sorted(used)

def renumber_citations(text, mapping):
    def _rewrite(match):
        raw = match.group(1)
        out = []
        for part in re.split(r'\s*,\s*', raw):
            if re.search(r'[-–]', part):
                a, b = re.split(r'[-–]', part)
                rng = range(int(a), int(b) + 1)
                mapped = [str(mapping.get(i, i)) for i in rng]
                # compress back to a range if contiguous
                try:
                    nums = list(map(int, mapped))
                    if nums == list(range(nums[0], nums[-1] + 1)):
                        out.append(f"{nums[0]}–{nums[-1]}")
                    else:
                        out.extend(map(str, mapped))
                except ValueError:
                    out.extend(mapped)
            else:
                i = int(part)
                out.append(str(mapping.get(i, i)))
        return "[" + ", ".join(out) + "]"
    return _CITE_RX.sub(_rewrite, text or "")