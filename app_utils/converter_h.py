
from __future__ import annotations
import re, json
from typing import Dict, Any, List, Tuple

# ---------------- Defaults ----------------
DEFAULTS = {
    "dropwise_timer_minutes": 10,          # default time if prose says "dropwise"/"slowly"
    "base_addition_ph_range": (9.0, 10.0), # default pH if base addition lacks explicit pH
    "ph_check_interval_minutes": 2,        # minutes between pH checks during addition
    "stir_idle_lookahead_ops": 8,          # how many ops we look ahead before auto-stopping stirrer
}

# Optional name normalization 
NAME_MAP = {
    "Flask 100 mL (FeCl3)": "Flask 100 mL (FeCl3·6H2O solution)",
}

# ---------------- Regex ----------------
TIME_RX = re.compile(r"(?:This\s+should\s+take|for|over|approximately|about)\s*(?:approximately\s*)?(\d+(?:\.\d+)?)\s*(minutes?|hours?)", re.I)
PH_RX   = re.compile(r"pH\s*(?:of(?:\s*around)?|around|=|≈|~)?\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|\bto\b)\s*(\d+(?:\.\d+)?)", re.I)

CONTAINER_KEYWORDS = ("flask","beaker","tube","vial","bottle","setup","conical","round-bottom","rbf","erlenmeyer")

# ---------------- Small helpers ----------------
def _norm(s: Any) -> Any:
    return s.strip() if isinstance(s, str) else s

def _is_container(name: str|None) -> bool:
    return isinstance(name, str) and any(k in name.lower() for k in CONTAINER_KEYWORDS)

def _contains_timer(ops: List[Dict[str,Any]]|None) -> bool:
    return any(o.get("op")=="timer" for o in (ops or []))

def _contains_monitor_ph(ops: List[Dict[str,Any]]|None) -> bool:
    return any(o.get("op")=="monitor_ph" for o in (ops or []))

def _has_raw_kw(step: Dict[str,Any], *keywords: str) -> bool:
    raw = (step.get("raw") or "").lower()
    return any(k.lower() in raw for k in keywords)

# ---------------- Holding rules ----------------
def ensure_pickup_before_action(micro_ops: List[Dict[str,Any]]|None) -> List[Dict[str,Any]]|None:
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

        # PLACE: must be holding the container
        if verb == "place" and obj and _is_container(obj) and obj not in held:
            out.append({"verb":"pick_up","object":obj,"from":location.get(obj,"bench")})
            held.add(obj)

        # POUR: must be holding source container
        if verb == "pour":
            src = obj if (obj and _is_container(obj) and op.get("from") is None) else (frm if _is_container(frm) else None)
            if src and src not in held:
                out.append({"verb":"pick_up","object":src,"from":location.get(src,"bench")})
                held.add(src)

        if verb == "pick_up" and obj:
            if op.get("from") in (None, "bench"):
                op["from"] = location.get(obj, "bench")
            held.add(obj); location[obj] = "hand"

        out.append(op)

        if verb == "place" and obj:
            held.discard(obj); location[obj] = to or location.get(obj,"bench")
    return out

# ---------------- Stir idle auto-stop ----------------
def inject_stir_stop_if_idle(micro_ops: List[Dict[str,Any]]|None, lookahead:int) -> List[Dict[str,Any]]|None:
    if not isinstance(micro_ops, list):
        return micro_ops
    out = []
    for i, op in enumerate(micro_ops):
        out.append(op)
        if isinstance(op, dict) and op.get("verb") == "wait":
            ahead = micro_ops[i+1:i+1+lookahead]
            will_mix = any(
                isinstance(a, dict) and (
                    (a.get("verb")=="set" and a.get("device")=="stir_plate" and a.get("param")=="rpm") or
                    (a.get("verb")=="pour") or
                    (a.get("verb")=="place" and a.get("to")=="stir_plate")
                ) for a in ahead
            )
            if not will_mix:
                out.append({"verb":"set","device":"stir_plate","param":"rpm","value":0,"note":"auto-stop to avoid idle spinning"})
    return out

# ---------------- Time & pH extraction + defaults ----------------
def _extract_minutes_from_raw(raw: str) -> int|None:
    m = TIME_RX.search(raw or "")
    if not m: return None
    val, unit = m.groups()
    return int(float(val)*60) if unit.lower().startswith("hour") else int(float(val))

def _extract_ph_range_from_raw(raw: str) -> Tuple[float,float]|None:
    m = PH_RX.search(raw or "")
    if not m: return None
    a, b = float(m.group(1)), float(m.group(2))
    return (min(a,b), max(a,b))

def _maybe_add_timer(step: Dict[str,Any], minutes: int) -> None:
    step["minutes"] = minutes
    step.setdefault("ops", [])
    if not _contains_timer(step["ops"]):
        step["ops"].append({"op":"timer","minutes":minutes})
    # Sync any existing waits
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
    # Add periodic pH checks into micro_ops during the timed window, or at least two
    step.setdefault("micro_ops", [])
    minutes = step.get("minutes", 0) or DEFAULTS["dropwise_timer_minutes"]
    interval = max(1, int(DEFAULTS["ph_check_interval_minutes"]))
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

        # Time
        minutes = _extract_minutes_from_raw(raw)
        if minutes is None and action in {"add","add_solvent","transfer"} and ("dropwise" in raw.lower() or "slowly" in raw.lower()):
            minutes = DEFAULTS["dropwise_timer_minutes"]
        if minutes is not None:
            _maybe_add_timer(step, minutes)

        # pH
        ph_range = _extract_ph_range_from_raw(raw)
        is_base_add = action in {"add","add_solvent","transfer"} and any(k in raw.lower() for k in ("naoh","sodium hydroxide","nh4oh","ammonium hydroxide","base"))
        if ph_range is None and is_base_add:
            ph_range = DEFAULTS["base_addition_ph_range"]
        if ph_range is not None and action in {"add","add_solvent","transfer"}:
            _maybe_add_monitor_ph(step, *ph_range)

# ---------------- Name normalization (optional) ----------------
def normalize_names(data: Dict[str,Any]) -> None:
    def repl(x: Any) -> Any:
        if isinstance(x, str) and x in NAME_MAP:
            return NAME_MAP[x]
        return x
    def walk(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: walk(repl(v)) for k,v in o.items()}
        if isinstance(o, list):
            return [walk(x) for x in o]
        return repl(o)
    normalized = walk(data)
    data.clear(); data.update(normalized)
    # registry touch-up example
    vr = data.get("vessel_registry")
    if isinstance(vr, dict):
        for k,v in list(vr.items()):
            if v == "FeCl3":
                vr[k] = "FeCl3·6H2O solution"

BASE_VERBS = {"pick_up","place","pour","set","wait"}

def _ensure_transfer_pours(step):
    """For any op.transfer in step.ops, inject a pour into step.micro_ops if missing."""
    ops = step.get("ops") or []
    mops = step.get("micro_ops") or []
    idx = mops[0].get("step_index") if mops else step.get("index")
    existing_pour = any(m.get("verb") == "pour" for m in mops)
    transfers = [op for op in ops if op.get("op") == "transfer"]
    # If any transfer exists and no pour recorded, add a single pour per transfer (conservative order: after placement)
    if transfers:
        # Find insertion point: after last 'place' or at end
        insert_at = 0
        for i, m in enumerate(mops):
            if m.get("verb") == "place":
                insert_at = i + 1
        for t in transfers:
            src = t.get("from") or "source"
            dst = t.get("to") or step.get("vessel") or step.get("target_vessel") or "V?"
            pour = {"verb":"pour","from": src, "to": dst, "step_index": idx}
            mops.insert(insert_at, pour)
            insert_at += 1
    step["micro_ops"] = mops

def _ensure_centrifuge_motion(step):
    """If step has a centrifuge op, ensure motion (pick_up→place CF1→wait→off→place back) exists in micro_ops."""
    ops = step.get("ops") or []
    if not any(op.get("op") == "centrifuge" for op in ops):
        return
    mops = step.get("micro_ops") or []
    idx = mops[0].get("step_index") if mops else step.get("index")
    tube = None
    # Find tube name from ops or a tube field
    for op in ops:
        if op.get("op") == "centrifuge":
            tube = op.get("tube") or step.get("tube") or "V2_tube"
            rpm  = op.get("rpm")
            mins = op.get("minutes") or step.get("minutes")  # priority to op minutes
            cfid = op.get("centrifuge_id") or "CF1"
            break
    # Presence checks
    has_pick = any(m.get("verb")=="pick_up" and m.get("object")==tube for m in mops)
    has_place_cf = any(m.get("verb")=="place" and m.get("to")=="centrifuge" for m in mops) or any(m.get("verb")=="place" and m.get("to")==cfid for m in mops)
    has_wait = any(m.get("verb")=="wait" for m in mops)
    # Build missing pieces conservatively at the start of micro_ops
    new = []
    if not has_pick:
        new.append({"verb":"pick_up","object": tube, "from": "rack", "step_index": idx})
    if not has_place_cf:
        new.append({"verb":"place","object": tube, "to": cfid, "step_index": idx})
    # ensure rpm + power on
    new.append({"verb":"set","device": cfid, "param":"rpm","value": rpm, "step_index": idx})
    new.append({"verb":"set","device": cfid, "param":"power","value": "on", "step_index": idx})
    # ensure wait with correct minutes
    if mins is not None:
        new.append({"verb":"wait","minutes": mins, "step_index": idx})
    # power off & place back
    new.append({"verb":"set","device": cfid, "param":"power","value": "off", "step_index": idx})
    new.append({"verb":"place","object": tube, "to": "rack", "step_index": idx})
    # Prepend new motions
    mops = new + mops
    # Normalize ANY existing waits in this step to 'mins' if present
    if mins is not None:
        norm = []
        for mm in mops:
            if mm.get('verb')=='wait':
                mm = {**mm, 'minutes': mins}
            norm.append(mm)
        mops = norm
    # Deduplicate consecutive identical entries
    dedup = []
    seen = set()
    for m in mops:
        key = json.dumps(m, sort_keys=True)
        if not dedup or json.dumps(dedup[-1], sort_keys=True) != key:
            dedup.append(m)
    step["micro_ops"] = dedup

def _sync_step_wait_minutes(step):
    """Sync any wait in step.micro_ops to step.minutes when present (unless centrifuge op provided its own)."""
    if step.get("minutes") is None:
        return
    minutes = step.get("minutes")
    new = []
    for m in step.get("micro_ops") or []:
        if m.get("verb") == "wait" and "minutes" in m and ("centrifuge" not in step.get("raw","")) and not any(op.get("op")=="centrifuge" for op in step.get("ops",[])):
            m = {**m, "minutes": minutes}
        new.append(m)
    step["micro_ops"] = new

def _unify_heating_placement(step):
    """If this step sets HP1.temperature_C but vessel is placed on SP1 only, switch placement target to HP1."""
    has_hp_temp = any(m.get("verb")=="set" and m.get("device") in ("HP1","hotplate","hotplate_id") and m.get("param")=="temperature_C" for m in step.get("micro_ops") or [])
    if not has_hp_temp:
        return
    for m in step.get("micro_ops") or []:
        if m.get("verb")=="place" and m.get("to")=="SP1":
            m["to"] = "HP1"

def _add_missing_vessels_to_registry(doc):
    reg = doc.setdefault("vessel_registry", {})
    known = set(reg.keys())
    referenced = set()
    # collect V* references
    for m in doc.get("micro_plan", []) or []:
        for k in ("object","from","to"):
            v = m.get(k)
            if isinstance(v,str) and re.match(r"^V\d+(_tube)?$", v):
                referenced.add(v)
    for st in doc.get("steps", []) or []:
        for m in st.get("micro_ops") or []:
            for k in ("object","from","to"):
                v = m.get(k)
                if isinstance(v,str) and re.match(r"^V\d+(_tube)?$", v):
                    referenced.add(v)
        for k in ("vessel","source_vessel","target_vessel","tube"):
            v = st.get(k)
            if isinstance(v,str) and re.match(r"^V\d+(_tube)?$", v):
                referenced.add(v)
    for v in sorted(referenced):
        if v not in known:
            reg[v] = "(auto) vessel"

def polish_robot_doc(doc):
    # operate on steps
    for st in doc.get("steps", []) or []:
        _ensure_transfer_pours(st)
        _ensure_centrifuge_motion(st)
        _sync_step_wait_minutes(st)
        _unify_heating_placement(st)
    _add_missing_vessels_to_registry(doc)
    # Enforce base verbs globally (be safe)
    allowed = BASE_VERBS
    doc["micro_plan"] = [m for m in (doc.get("micro_plan") or []) if m.get("verb") in allowed]
    for st in doc.get("steps", []) or []:
        st["micro_ops"] = [m for m in (st.get("micro_ops") or []) if m.get("verb") in allowed]
    return doc

# ---------------- Public entry ----------------
def apply_postprocessing(data: Dict[str,Any]) -> Dict[str,Any]:
    """
    Postprocess a converter result dict in-place and return it.
    - Inserts missing pick_up before place/pour
    - Adds timers and pH monitoring from prose; applies defaults when absent
    - Auto-stops idle stir plates after waits
    - Optionally normalizes vessel names
    """
    if not isinstance(data, dict):
        return data

    # Optional: normalize names first so downstream logic sees consistent labels
    normalize_names(data)

    # Step-level augmentations (time/pH + defaults)
    augment_steps_with_time_ph_and_defaults(data)

    # Global micro_plan passes
    lookahead = int(DEFAULTS["stir_idle_lookahead_ops"])
    if isinstance(data.get("micro_plan"), list):
        data["micro_plan"] = ensure_pickup_before_action(data["micro_plan"])
        data["micro_plan"] = inject_stir_stop_if_idle(data["micro_plan"], lookahead=lookahead)

    # Per-step micro_ops passes
    for step in data.get("steps", []) or []:
        if isinstance(step, dict):
            step["micro_ops"] = ensure_pickup_before_action(step.get("micro_ops", []))
            step["micro_ops"] = inject_stir_stop_if_idle(step["micro_ops"], lookahead=lookahead)

    return data

# ---------------- CLI (optional) ----------------
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Postprocess a converter JSON with holding/time/pH/stir fixes.")
    ap.add_argument("-i","--input", required=True)
    ap.add_argument("-o","--output", required=True)
    args = ap.parse_args()
    with open(args.input,"r",encoding="utf-8") as f:
        obj = json.load(f)
    obj = apply_postprocessing(obj)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")
