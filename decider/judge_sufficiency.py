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

def judge_sufficiency(hits: List[dict], intent: str):
    k = min(MAX_K, len(hits))
    top = hits[:k]

    top1 = clamp01(top[0]["sim"]) if k else 0.0
    mean_topk = clamp01(sum(h["sim"] for h in top) / k) if k else 0.0
    evidence_count = sum((h["sim"] or 0.0) >= SIM_FLOOR for h in top)
    distinct_sources = len(set(h.get("source_domain","") for h in top if h.get("source_domain")))

    merged_slots = set()
    entity_ok = False
    for h in top:
        merged_slots |= set(h.get("slots_present", []))
        entity_ok = entity_ok or bool(h.get("entity_hit", False))

    required = REQUIRED_BY_INTENT.get(intent, {"MATERIAL"}) or {"MATERIAL"}
    coverage = len(merged_slots & required) / float(len(required))

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
