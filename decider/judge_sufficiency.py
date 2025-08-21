from __future__ import annotations
from typing import List, Dict, Set, Tuple

SIM_FLOOR = 0.40
MAX_K = 5

REQUIRED_BY_INTENT: Dict[str, Set[str]] = {
    "procedure": {"ACTION","MATERIAL","AMOUNT","TEMP","TIME","EQUIPMENT","ATMOS"},
    "definition": {"MATERIAL"},
    "comparison": {"MATERIAL","METRIC","CONDITION"},
    "mechanism": {"MATERIAL","ACTION","CONDITION"},
}

THRESHOLD: Dict[str, float] = {
    "procedure": 0.60,
    "definition": 0.45,
    "comparison": 0.55,
    "mechanism": 0.55,
}

def _fresh_ok(hits: List[dict], intent: str) -> bool:
    newest_year = max((h.get("year", 0) or 0) for h in hits) if hits else 0
    if intent in ("comparison",):
        return newest_year >= 2022
    return True

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def judge_sufficiency(hits, intent: str):
    """
    Robust version: accepts
      - list[dict] (normal case)
      - list[str]  (JSON lines or raw strings)
      - str path to a .jsonl file
      - str containing JSON (array or object)
    and coerces everything into a list[dict] with the keys we score on.
    """
    import json, os

    def _iter_as_dicts(obj):
        # If a PATH to .jsonl
        if isinstance(obj, str) and os.path.exists(obj):
            with open(obj, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        d = json.loads(s)
                    except Exception:
                        d = None
                    yield d if isinstance(d, dict) else {"text": s}
            return

        # If a JSON string (object/array)
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
                if isinstance(parsed, dict):
                    yield parsed; return
                if isinstance(parsed, list):
                    for it in parsed:
                        if isinstance(it, dict):
                            yield it
                        elif isinstance(it, str):
                            try:
                                d = json.loads(it)
                                yield d if isinstance(d, dict) else {"text": it}
                            except Exception:
                                yield {"text": it}
                    return
            except Exception:
                # plain text string fallback
                yield {"text": obj}; return

        # Iterable of items
        try:
            for it in obj:
                if isinstance(it, dict):
                    yield it
                elif isinstance(it, str):
                    try:
                        d = json.loads(it)
                        yield d if isinstance(d, dict) else {"text": it}
                    except Exception:
                        yield {"text": it}
        except TypeError:
            # Not iterable; nothing to yield
            return

    def _to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    # ---- Normalize hits into dicts with safe defaults ----
    norm = []
    for d in _iter_as_dicts(hits):
        if not isinstance(d, dict):
            continue
        norm.append({
            "sim": _to_float(d.get("sim", 0.0), 0.0),
            "source_domain": d.get("source_domain", "") or "",
            "slots_present": list(d.get("slots_present") or []),
            "entity_hit": bool(d.get("entity_hit", False)),
            # pass through anything else (timestamps, etc.) for _fresh_ok
            **{k: v for k, v in d.items() if k not in {"sim","source_domain","slots_present","entity_hit"}}
        })

    # ---- Original scoring logic (unchanged, but safer gets) ----
    k = min(MAX_K, len(norm))
    top = norm[:k]

    top1 = clamp01(top[0]["sim"]) if k else 0.0
    mean_topk = clamp01(sum(_to_float(h.get("sim", 0.0), 0.0) for h in top) / k) if k else 0.0
    evidence_count = sum((_to_float(h.get("sim", 0.0), 0.0)) >= SIM_FLOOR for h in top)
    distinct_sources = len({(h.get("source_domain") or "") for h in top if h.get("source_domain")})

    merged_slots, entity_ok = set(), False
    for h in top:
        merged_slots |= set(h.get("slots_present") or [])
        entity_ok = entity_ok or bool(h.get("entity_hit"))

    required = REQUIRED_BY_INTENT.get(intent, {"MATERIAL"}) or {"MATERIAL"}
    coverage = (len(merged_slots & required) / float(len(required))) if required else 0.0

    fresh_ok = _fresh_ok(top, intent)

    score = (
        0.30*top1 +
        0.20*mean_topk +
        0.20*coverage +
        0.10*(evidence_count/3.0) +
        0.10*(distinct_sources/3.0) +
        0.10*(1.0 if fresh_ok else 0.0)
    )
    score = clamp01(score)

    if k == 0:
        return 0.0, "mine", {"reason":"no_hits","top1":top1,"mean_topk":mean_topk,"coverage":coverage,"fresh_ok":fresh_ok,"entity_ok":False,"evidence_count":evidence_count,"distinct_sources":distinct_sources}
    if not entity_ok:
        return 0.0, "mine", {"reason":"no_entity","top1":top1,"mean_topk":mean_topk,"coverage":coverage,"fresh_ok":fresh_ok,"entity_ok":False,"evidence_count":evidence_count,"distinct_sources":distinct_sources}
    if intent == "procedure" and coverage < 0.5:
        return 0.0, "mine", {"reason":"low_coverage","top1":top1,"mean_topk":mean_topk,"coverage":coverage,"fresh_ok":fresh_ok,"entity_ok":entity_ok,"evidence_count":evidence_count,"distinct_sources":distinct_sources}

    decision = "use_kb" if score >= THRESHOLD.get(intent, 0.5) else "mine"
    return score, decision, {"reason":"threshold" if decision=="mine" else "ok","top1":top1,"mean_topk":mean_topk,"coverage":coverage,"fresh_ok":fresh_ok,"entity_ok":entity_ok,"evidence_count":evidence_count,"distinct_sources":distinct_sources,"threshold":THRESHOLD.get(intent,0.5),"score":score}