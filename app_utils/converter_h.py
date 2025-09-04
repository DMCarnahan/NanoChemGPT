from __future__ import annotations
import re, json, argparse
from typing import Dict, Any, List

# ------------------------ heuristics ------------------------
CONTAINER_KEYWORDS = ("flask","beaker","tube","vial","bottle","setup","conical","round-bottom","rbf","erlenmeyer")
TIME_RX = re.compile(r"(?:This\s+should\s+take|for|over|approximately|about)\s*(?:approximately\s*)?(\d+(?:\.\d+)?)\s*(minutes?|hours?)", re.I)
PH_RX   = re.compile(r"pH\s*(?:of(?:\s*around)?|around|=|≈|~)?\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|\bto\b)\s*(\d+(?:\.\d+)?)", re.I)

def _norm(s: Any) -> Any:
    return s.strip() if isinstance(s, str) else s

def _is_container(name: str|None) -> bool:
    if not isinstance(name, str): return False
    low = name.lower()
    return any(k in low for k in CONTAINER_KEYWORDS)

def _contains_timer(ops: List[Dict[str,Any]]) -> bool:
    return any(o.get("op")=="timer" for o in ops or [])

def _contains_monitor_ph(ops: List[Dict[str,Any]]) -> bool:
    return any(o.get("op")=="monitor_ph" for o in ops or [])

# ------------------------ holding logic ------------------------
def ensure_pickup_before_action(micro_ops: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """Insert 'pick_up' before 'place' or 'pour' requiring a handheld container."""
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

        # PLACE needs object in hand if it's a container
        if verb == "place" and obj and _is_container(obj) and obj not in held:
            out.append({"verb":"pick_up","object":obj,"from":location.get(obj,"bench")})
            held.add(obj)

        # POUR often encodes source as 'from' or 'object'
        if verb == "pour":
            src = obj if (obj and _is_container(obj) and op.get("from") is None) else (frm if _is_container(frm) else None)
            if src and src not in held:
                out.append({"verb":"pick_up","object":src,"from":location.get(src,"bench")})
                held.add(src)

        if verb == "pick_up" and obj:
            held.add(obj); location[obj] = "hand"

        out.append(op)

        if verb == "place" and obj:
            held.discard(obj); location[obj] = to or location.get(obj,"bench")
    return out

# ------------------------ idle stir-plate logic ------------------------
def inject_stir_stop_if_idle(micro_ops: List[Dict[str,Any]], lookahead:int=8) -> List[Dict[str,Any]]:
    """If a 'wait' is followed by no mixing/transfer soon, stop stir plate (rpm->0)."""
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

# ------------------------ time & pH augmentation ------------------------
def _augment_step_from_raw(step: Dict[str,Any]) -> None:
    raw = step.get("raw","") or ""
    # time -> minutes + timer op
    mt = TIME_RX.search(raw)
    if mt:
        val, unit = mt.groups()
        minutes = int(float(val)*60) if unit.lower().startswith("hour") else int(float(val))
        step["minutes"] = minutes
        step.setdefault("ops", [])
        if not _contains_timer(step["ops"]):
            step["ops"].append({"op":"timer","minutes":minutes})

    # pH range -> monitor_ph + micro checks (for add/transfer only)
    mph = PH_RX.search(raw)
    if mph and step.get("action") in {"add","transfer"}:
        lo, hi = sorted(map(float, mph.groups()))
        step.setdefault("ops", [])
        if not _contains_monitor_ph(step["ops"]):
            step["ops"].append({
                "op":"monitor_ph","target_range":[lo,hi],"sensor":"pH_probe","strategy":"titrate_addition",
                "notes":"Maintain pH during dropwise addition"
            })
        checks = max(2, step.get("minutes", 6)//2)
        step.setdefault("micro_ops", [])
        # Avoid over-duplicating pH checks if already present
        existing = sum(1 for m in step["micro_ops"] if isinstance(m,dict) and m.get("verb")=="measure" and m.get("param")=="pH")
        to_add = max(0, checks - existing)
        for _ in range(to_add):
            step["micro_ops"].append({"verb":"measure","param":"pH"})
            step["micro_ops"].append({"verb":"set","param":"addition_rate","value":"slow"})

def augment_steps_with_time_ph(data: Dict[str,Any]) -> None:
    for step in data.get("steps", []) or []:
        if isinstance(step, dict):
            _augment_step_from_raw(step)

# ------------------------ name normalization ------------------------
def normalize_names(data: Dict[str,Any]) -> None:
    # Customize this with whatever aliases you see in your runs
    replacements = {
        "Flask 100 mL (FeCl3)": "Flask 100 mL (FeCl3·6H2O solution)",
    }
    def repl(x: Any) -> Any:
        if isinstance(x, str) and x in replacements:
            return replacements[x]
        return x
    def walk(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: walk(repl(v)) for k,v in o.items()}
        if isinstance(o, list):
            return [walk(x) for x in o]
        return repl(o)
    # mutate in place
    normalized = walk(data)
    data.clear(); data.update(normalized)

    # vessel registry touch-up
    vr = data.get("vessel_registry")
    if isinstance(vr, dict):
        for k,v in list(vr.items()):
            if v == "FeCl3":
                vr[k] = "FeCl3·6H2O solution"

# ------------------------ main entry ------------------------
def apply_postprocessing(data: Dict[str,Any]) -> Dict[str,Any]:
    """Apply all postprocessing passes and return the dict (mutates in place)."""
    if not isinstance(data, dict):
        return data
    # names first so micro_plan ops see normalized names
    normalize_names(data)
    # time/pH augmentation
    augment_steps_with_time_ph(data)
    # micro_plan passes
    if isinstance(data.get("micro_plan"), list):
        data["micro_plan"] = ensure_pickup_before_action(data["micro_plan"])
        data["micro_plan"] = inject_stir_stop_if_idle(data["micro_plan"], lookahead=8)
    return data

# ------------------------ CLI ------------------------
def _main():
    ap = argparse.ArgumentParser(description="Enhance converter output with holding, time/pH, and stir-plate fixes.")
    ap.add_argument("-i","--input", required=True, help="Input JSON file (converter output)")
    ap.add_argument("-o","--output", required=True, help="Output JSON file")
    args = ap.parse_args()

    with open(args.input,"r",encoding="utf-8") as f:
        data = json.load(f)
    apply_postprocessing(data)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    _main()
