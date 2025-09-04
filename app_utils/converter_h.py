from __future__ import annotations
import re, json
from typing import Dict, Any, List, Tuple

CONTAINER_KEYWORDS = ("flask","beaker","tube","vial","bottle","setup","conical","round-bottom","rbf","erlenmeyer")
TIME_RX = re.compile(r"(?:This\s+should\s+take|for|over|approximately|about)\s*(?:approximately\s*)?(\d+(?:\.\d+)?)\s*(minutes?|hours?)", re.I).pattern
PH_RX   = re.compile(r"pH\s*(?:of(?:\s*around)?|around|=|≈|~)?\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|\bto\b)\s*(\d+(?:\.\d+)?)", re.I).pattern

# Recompile with single backslashes by decoding the string literal escapes
TIME_RX = re.compile("(?:This\s+should\s+take|for|over|approximately|about)\s*(?:approximately\s*)?(\d+(?:\.\d+)?)\s*(minutes?|hours?)", re.I)
PH_RX   = re.compile("pH\s*(?:of(?:\s*around)?|around|=|≈|~)?\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|\bto\b)\s*(\d+(?:\.\d+)?)", re.I)

DEFAULTS = {
    "dropwise_timer_minutes": 10,
    "base_addition_ph_range": (9.0, 10.0),
    "ph_check_interval_minutes": 2,
    "stir_idle_lookahead_ops": 8,
}

NAME_MAP = {
    "Flask 100 mL (FeCl3)": "Flask 100 mL (FeCl3·6H2O solution)",
}

def _norm(s: Any) -> Any:
    return s.strip() if isinstance(s, str) else s

def _is_container(name: str|None) -> bool:
    return isinstance(name, str) and any(k in name.lower() for k in CONTAINER_KEYWORDS)

def _contains_timer(ops: List[Dict[str,Any]]) -> bool:
    return any(o.get("op")=="timer" for o in (ops or []))

def _contains_monitor_ph(ops: List[Dict[str,Any]]) -> bool:
    return any(o.get("op")=="monitor_ph" for o in (ops or []))

def _has_raw_kw(step: Dict[str,Any], *keywords: str) -> bool:
    raw = (step.get("raw") or "").lower()
    return any(k.lower() in raw for k in keywords)

def ensure_pickup_before_action(micro_ops: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not isinstance(micro_ops, list):
        return micro_ops
    held, location, out = set(), {}, []
    for op in micro_ops:
        if not isinstance(op, dict):
            out.append(op); continue
        op = dict(op)
        verb = op.get("verb")
        obj  = _norm(op.get("object"))
        frm  = _norm(op.get("from"))
        to   = _norm(op.get("to"))
        if obj and obj not in location:
            location[obj] = "bench"
        if verb == "place" and obj and _is_container(obj) and obj not in held:
            out.append({"verb":"pick_up","object":obj,"from":location.get(obj,"bench")}); held.add(obj)
        if verb == "pour":
            src = obj if (obj and _is_container(obj) and op.get("from") is None) else (frm if _is_container(frm) else None)
            if src and src not in held:
                out.append({"verb":"pick_up","object":src,"from":location.get(src,"bench")}); held.add(src)
        if verb == "pick_up" and obj:
            if op.get("from") in (None, "bench"):
                op["from"] = location.get(obj, "bench")
            held.add(obj); location[obj] = "hand"
        out.append(op)
        if verb == "place" and obj:
            held.discard(obj); location[obj] = to or location.get(obj,"bench")
    return out

def inject_stir_stop_if_idle(micro_ops: List[Dict[str,Any]], lookahead:int) -> List[Dict[str,Any]]:
    if not isinstance(micro_ops, list):
        return micro_ops
    out = []
    for i, op in enumerate(micro_ops):
        out.append(op)
        if isinstance(op, dict) and op.get("verb") == "wait":
            ahead = micro_ops[i+1:i+1+lookahead]
            will_mix = any(
                isinstance(a, dict) and (
                    (a.get("verb")=="set" and a.get("device")=="stir_plate" and a.get("param")=="rpm" and a.get("value",0)) or
                    (a.get("verb")=="pour")
                ) for a in ahead
            )
            if not will_mix:
                out.append({"verb":"set","device":"stir_plate","param":"rpm","value":0,"note":"auto-stop to avoid idle spinning"})
    return out

def _extract_minutes_from_raw(raw: str):
    m = TIME_RX.search(raw or "")
    if not m: return None
    val, unit = m.groups()
    return int(float(val)*60) if unit.lower().startswith("hour") else int(float(val))

def _extract_ph_range_from_raw(raw: str):
    m = PH_RX.search(raw or "")
    if not m: return None
    a, b = float(m.group(1)), float(m.group(2))
    return (min(a,b), max(a,b))

def _maybe_add_timer(step: Dict[str,Any], minutes: int) -> None:
    step["minutes"] = minutes
    step.setdefault("ops", [])
    if not _contains_timer(step["ops"]):
        step["ops"].append({"op":"timer","minutes":minutes})
    for seq_key in ("ops","micro_ops"):
        seq = step.get(seq_key)
        if not isinstance(seq, list): continue
        for o in seq:
            if isinstance(o, dict) and (o.get("op")=="wait" or o.get("verb")=="wait"):
                o["minutes"] = minutes

def _maybe_add_monitor_ph(step: Dict[str,Any], lo: float, hi: float) -> None:
    step.setdefault("ops", [])
    if not _contains_monitor_ph(step["ops"]):
        step["ops"].append({
            "op":"monitor_ph","target_range":[lo,hi],"sensor":"pH_probe","strategy":"titrate_addition",
            "notes":"Maintain pH during addition"
        })
    step.setdefault("micro_ops", [])
    minutes = step.get("minutes", 0) or 10
    interval = max(1, 2)
    checks = max(2, minutes // interval)
    existing = sum(1 for m in step["micro_ops"] if isinstance(m,dict) and m.get("verb")=="measure" and m.get("param")=="pH")
    to_add = max(0, checks - existing)
    for _ in range(to_add):
        step["micro_ops"].append({"verb":"measure","param":"pH"})
        step["micro_ops"].append({"verb":"set","param":"addition_rate","value":"slow"})

def augment_steps_with_time_ph_and_defaults(data: Dict[str,Any]) -> None:
    for step in data.get("steps", []) or []:
        if not isinstance(step, dict): continue
        raw = step.get("raw","") or ""
        action = (step.get("action") or "").lower()
        minutes = _extract_minutes_from_raw(raw)
        ph_range = _extract_ph_range_from_raw(raw)
        if minutes is None and action in {"add","add_solvent","transfer"} and ("dropwise" in raw.lower() or "slowly" in raw.lower()):
            minutes = 10
        if ph_range is None and action in {"add","add_solvent","transfer"} and any(k in raw.lower() for k in ("naoh","sodium hydroxide","nh4oh","ammonium hydroxide","base")):
            ph_range = (9.0, 10.0)
        if minutes is not None: _maybe_add_timer(step, minutes)
        if ph_range is not None: _maybe_add_monitor_ph(step, *ph_range)

def normalize_names(data: Dict[str,Any]) -> None:
    def repl(x: Any) -> Any:
        if isinstance(x, str) and x in NAME_MAP: return NAME_MAP[x]
        return x
    def walk(o: Any) -> Any:
        if isinstance(o, dict): return {k: walk(repl(v)) for k,v in o.items()}
        if isinstance(o, list): return [walk(x) for x in o]
        return repl(o)
    normalized = walk(data)
    data.clear(); data.update(normalized)
    vr = data.get("vessel_registry")
    if isinstance(vr, dict):
        for k,v in list(vr.items()):
            if v == "FeCl3":
                vr[k] = "FeCl3·6H2O solution"

def apply_postprocessing(data: Dict[str,Any]) -> Dict[str,Any]:
    if not isinstance(data, dict): return data
    normalize_names(data)
    augment_steps_with_time_ph_and_defaults(data)
    lookahead = 8
    if isinstance(data.get("micro_plan"), list):
        data["micro_plan"] = ensure_pickup_before_action(data["micro_plan"])
        data["micro_plan"] = inject_stir_stop_if_idle(data["micro_plan"], lookahead=lookahead)
    for step in data.get("steps", []) or []:
        if isinstance(step, dict):
            step["micro_ops"] = ensure_pickup_before_action(step.get("micro_ops", []))
            step["micro_ops"] = inject_stir_stop_if_idle(step.get("micro_ops", []), lookahead=lookahead)
    return data
