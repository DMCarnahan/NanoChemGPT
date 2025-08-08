
import re, json, pathlib
from typing import List, Dict, Optional, Tuple

DEFAULTS = {
    "stir_rpm": 700,
    "centrifuge_rpm": 4000,
    "centrifuge_minutes": 10,
    "transfer_rate_slow": "slow",
    "room_temp_C": 25,
    "default_transfer_volume_mL": None  # if not stated, leave None
}

DEVICE_IDS = {
    "stir_plate_id": "SP1",
    "hotplate_id": "HP1",
    "centrifuge_id": "CF1",
    "oven_id": "OV1",
}

TAG_RX = re.compile(r"\s*\[(?:CTX|DB|PARSED|GEN|\d+)\]\s*$")

def strip_tags(s: str) -> str:
    return TAG_RX.sub("", s).strip()

def find_temp_c(t: str):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*C\b", t, re.I)
    return float(m.group(1)) if m else None

def find_minutes(t: str):
    mins = 0.0; found = False
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:hour|hr|hrs|h)\b", t, re.I):
        mins += float(m.group(1)) * 60; found = True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:minute|min|mins|m)\b", t, re.I):
        mins += float(m.group(1)); found = True
    return mins if found else None

class VesselRegistry:
    def __init__(self):
        self._name_to_vid = {}; self._vid_to_name = {}; self._counter = 0
        self.primary_vessel: Optional[str] = None
    def _new_vid(self):
        self._counter += 1; return f"V{self._counter}"
    def ensure(self, name: str) -> str:
        key = name.lower().strip()
        if key not in self._name_to_vid:
            vid = self._new_vid()
            self._name_to_vid[key] = vid; self._vid_to_name[vid] = name
            if self.primary_vessel is None: self.primary_vessel = vid
        return self._name_to_vid[key]
    def lookup(self, name: str):
        return self._name_to_vid.get(name.lower().strip())
    def as_dict(self):
        return {vid: name for vid, name in self._vid_to_name.items()}

# ---------------- Extended concentration parsing ----------------

def parse_concentration_block(s: str) -> Optional[Dict]:
    s_clean = s
    # 1) molarity
    m = re.search(r"(\d+(?:\.\d+)?)\s*([mM])\b", s_clean)
    if m:
        return {"concentration": float(m.group(1)), "concentration_units": "M", "kind": "molar"}
    # 2) percent forms: % w/v, % w/w, % v/v
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(w\/v|w\/w|v\/v)\b", s_clean, re.I)
    if m:
        return {"concentration": float(m.group(1)), "concentration_units": f"% {m.group(2).lower()}", "kind": "percent"}
    # 3) mg/mL or g/L
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg\/mL|g\/L)\b", s_clean, re.I)
    if m:
        unit = m.group(2)
        return {"concentration": float(m.group(1)), "concentration_units": unit, "kind": "mass_per_volume"}
    return None

def _clean_solvent_tail(solvent: str) -> str:
    solvent = strip_tags(solvent.strip().rstrip(",."))
    solvent = solvent.split(" in ")[0].strip()
    return solvent

# ---------------- Detection: solution prep to dispense ----------------

def detect_solution_prep(line: str) -> Optional[Dict]:
    s = strip_tags(line.strip().rstrip("."))

    # Generic concentration finder for the line
    conc_info = parse_concentration_block(s)

    # Try multiple phrasings that include volume & solvent
    pats = [
        r"prepare\s+a\s+(?P<conc>[\d\.]+\s*(?:m|M|%(?:\s*(?:w\/v|w\/w|v\/v))?|(?:mg\/mL|g\/L)))\s+(?:.+?\s+)?solution\s+of\s+.+?\s+by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s*(?:in\b|$)",
        r"prepare\s+a\s+(?P<conc>[\d\.]+\s*(?:m|M|%(?:\s*(?:w\/v|w\/w|v\/v))?|(?:mg\/mL|g\/L)))\s+(?P<xname>.+?)\s+solution\s+by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s*(?:in\b|$)",
        r"dissolv\w*\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s+to\s+(?:make|form|yield|obtain)\s+a\s+(?P<conc>[\d\.]+\s*(?:m|M|%(?:\s*(?:w\/v|w\/w|v\/v))?|(?:mg\/mL|g\/L)))\s+.+?\s+solution",
        r"(?:add|charge)\s+(?P<solute>.+?)\s+to\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s+to\s+(?:make|form|yield|obtain)\s+a\s+(?P<conc>[\d\.]+\s*(?:m|M|%(?:\s*(?:w\/v|w\/w|v\/v))?|(?:mg\/mL|g\/L)))\s+.+?\s+solution",
        r"(?:make|formulate|compose)\s+a\s+(?P<conc>[\d\.]+\s*(?:m|M|%(?:\s*(?:w\/v|w\/w|v\/v))?|(?:mg\/mL|g\/L)))\s+.+?\s+solution\s+by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\b",
        r"(?:make|formulate|compose)\s+.+?\(\s*(?P<conc>[\d\.]+\s*(?:m|M|%(?:\s*(?:w\/v|w\/w|v\/v))?|(?:mg\/mL|g\/L)))\s*\)\s+by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\b",
    ]

    for pat in pats:
        m = re.search(pat, s, re.I)
        if m:
            conc_str = m.group("conc") if "conc" in m.groupdict() else None
            if conc_str:
                # Normalize concentration
                ci = parse_concentration_block(conc_str)
            else:
                ci = conc_info
            if not ci:
                continue
            solute = m.group("solute").strip()
            vol = float(m.group("vol")); vunit = m.group("vunit")
            solvent = _clean_solvent_tail(m.group("solvent"))
            return {
                "action": "dispense",
                "solute": solute,
                "solvent": solvent,
                "concentration": ci["concentration"],
                "concentration_units": ci["concentration_units"],
                "concentration_kind": ci["kind"],
                "volume": vol,
                "volume_units": vunit
            }
    return None

# ---------------- Add detection with volumes ----------------

def parse_volume_inline(s: str) -> Optional[Tuple[float, str]]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mL|ml|L|l)\b", s)
    if m:
        return (float(m.group(1)), m.group(2))
    return None

def detect_add_while_stirring(line: str) -> Optional[Dict]:
    s = strip_tags(line.strip())
    # with stirring
    m = re.search(r"\b(add|charge)\s+(?:the\s+)?(?P<src>.+?)\s+to\s+(?:the\s+)?(?P<dst>.+?)\s+(?:while\s+stirring|with\s+stirring|under\s+stirring)", s, re.I)
    if m:
        vol_info = parse_volume_inline(s)
        return {
            "action": "add",
            "source_name": m.group("src").strip(),
            "target_name": m.group("dst").strip(),
            "with_stirring": True,
            "rate": "slow" if ("slow" in s.lower() or "dropwise" in s.lower()) else "normal",
            "volume": vol_info[0] if vol_info else DEFAULTS["default_transfer_volume_mL"],
            "volume_units": vol_info[1] if vol_info else ("mL" if DEFAULTS["default_transfer_volume_mL"] else None)
        }
    # simple add
    m2 = re.search(r"\b(add|charge)\s+(?:the\s+)?(?P<src>.+?)\s+to\s+(?:the\s+)?(?P<dst>.+?)\b", s, re.I)
    if m2:
        vol_info = parse_volume_inline(s)
        return {
            "action": "add",
            "source_name": m2.group("src").strip(),
            "target_name": m2.group("dst").strip(),
            "with_stirring": False,
            "rate": "normal",
            "volume": vol_info[0] if vol_info else DEFAULTS["default_transfer_volume_mL"],
            "volume_units": vol_info[1] if vol_info else ("mL" if DEFAULTS["default_transfer_volume_mL"] else None)
        }
    return None

# ---------------- Stir/heat & post-processing (same as v1) ----------------

def detect_stir_then_heat(line: str):
    s = strip_tags(line.strip())
    if "stir" in s.lower() and "heat" in s.lower():
        durations = [float(x[0])*(60 if re.search(r"(?:hour|hr|hrs|h)\b", x[1], re.I) else 1) for x in re.findall(r"(\d+(?:\.\d+)?)\s*((?:hour|hr|hrs|h|minute|min|mins|m)\b)", s, re.I)]
        stir_minutes = durations[0] if durations else 60.0
        heat_minutes = durations[1] if len(durations)>1 else 60.0
        heat_temp = find_temp_c(s) or DEFAULTS["room_temp_C"]
        return [{"action":"stir","minutes":stir_minutes,"temperature_C":DEFAULTS["room_temp_C"]},
                {"action":"heat_hold","temperature_C":heat_temp,"minutes":heat_minutes}]
    if "stir" in s.lower():
        return [{"action":"stir","minutes":find_minutes(s) or 60.0,"temperature_C":DEFAULTS["room_temp_C"]}]
    if "heat" in s.lower():
        return [{"action":"heat_hold","temperature_C":find_temp_c(s) or DEFAULTS["room_temp_C"],"minutes":find_minutes(s) or 60.0}]
    return None

def detect_postproc(line: str):
    s = strip_tags(line.strip())
    if any(k in s.lower() for k in ["centrifuge","wash","dry","oven"]):
        ops = []
        if "cool" in s.lower():
            ops.append({"action":"cool_to_room"})
        if "centrifuge" in s.lower():
            ops.append({"action":"centrifuge","rpm":DEFAULTS["centrifuge_rpm"],"minutes":DEFAULTS["centrifuge_minutes"]})
            ops.append({"action":"decant_supernatant"})
        if "wash" in s.lower():
            wash_solvent = "deionized water" if ("deionized water" in s.lower() or "di water" in s.lower()) else "wash solvent"
            ops += [{"action":"add_wash_solvent","solvent":wash_solvent},{"action":"resuspend"},{"action":"centrifuge","rpm":DEFAULTS["centrifuge_rpm"],"minutes":DEFAULTS["centrifuge_minutes"]},{"action":"decant_supernatant"}]
        if "dry" in s.lower() or "oven" in s.lower():
            temp = find_temp_c(s) or 60.0; minutes = find_minutes(s) or 120.0
            ops.append({"action":"oven_dry","temperature_C":temp,"minutes":minutes})
        return ops if ops else None
    return None

# ---------------- Primitive ops ----------------

def ops_for_dispense(vessel, solute, solvent, volume_val, volume_unit):
    return [{"op":"ensure_vessel","vessel":vessel},
            {"op":"add_solute","vessel":vessel,"reagent":solute},
            {"op":"add_solvent","vessel":vessel,"solvent":solvent,"volume":volume_val,"volume_units":volume_unit}]

def ops_for_add(src_v, dst_v, with_stir, rate, vol, vol_units):
    ops=[]; 
    if with_stir:
        ops.append({"op":"move_to_stir_plate","vessel":dst_v,"stir_plate_id":DEVICE_IDS["stir_plate_id"]})
        ops.append({"op":"set_stir_rate","vessel":dst_v,"rpm":DEFAULTS["stir_rpm"]})
    op = {"op":"transfer","from":src_v,"to":dst_v,"rate":rate}
    if vol is not None:
        op["volume"] = vol; op["volume_units"] = vol_units or "mL"
    ops.append(op)
    return ops

def ops_for_stir(vessel, minutes, temp_C):
    return [{"op":"move_to_stir_plate","vessel":vessel,"stir_plate_id":DEVICE_IDS["stir_plate_id"]},
            {"op":"set_stir_rate","vessel":vessel,"rpm":DEFAULTS["stir_rpm"]},
            {"op":"wait","minutes":minutes}]

def ops_for_heat(vessel, temp_C, minutes):
    return [{"op":"set_hotplate_temperature","hotplate_id":DEVICE_IDS["hotplate_id"],"temperature_C":temp_C},
            {"op":"wait","minutes":minutes}]

def ops_for_postproc(vessel, actions):
    ops = []
    for a in actions:
        if a["action"]=="cool_to_room":
            ops.append({"op":"set_hotplate_temperature","hotplate_id":DEVICE_IDS["hotplate_id"],"temperature_C":DEFAULTS["room_temp_C"]})
        elif a["action"]=="centrifuge":
            ops.append({"op":"transfer_to_centrifuge_tube","from":vessel,"to":f"{vessel}_tube"})
            ops.append({"op":"centrifuge","centrifuge_id":DEVICE_IDS["centrifuge_id"],"rpm":a["rpm"],"minutes":a["minutes"]})
        elif a["action"]=="decant_supernatant":
            ops.append({"op":"decant_supernatant","tube":f"{vessel}_tube"})
        elif a["action"]=="add_wash_solvent":
            ops.append({"op":"add_wash_solvent","tube":f"{vessel}_tube","solvent":a["solvent"]})
        elif a["action"]=="resuspend":
            ops.append({"op":"resuspend","tube":f"{vessel}_tube"})
        elif a["action"]=="oven_dry":
            ops.append({"op":"move_to_oven","tube":f"{vessel}_tube","oven_id":DEVICE_IDS["oven_id"]})
            ops.append({"op":"set_oven_temperature","oven_id":DEVICE_IDS["oven_id"],"temperature_C":a["temperature_C"]})
            ops.append({"op":"wait","minutes":a["minutes"]})
    return ops

# ---------------- Extraction & Conversion ----------------

def extract_steps(markdown_text: str):
    lines = markdown_text.splitlines(); in_proc=False; steps=[]; buf=[]
    for line in lines:
        if re.match(r"\s*3\.\s*\*\*Procedure\*\*:", line): in_proc=True; continue
        if in_proc:
            if re.match(r"\s*\d+\.\s", line):
                if buf: steps.append(" ".join(buf).strip()); buf=[]
                buf.append(re.sub(r"^\s*\d+\.\s*", "", line).strip())
            else:
                if line.strip(): buf.append(line.strip())
    if buf: steps.append(" ".join(buf).strip())
    return [strip_tags(s) for s in steps if s.strip()]

def convert_text_to_robot_ops(text: str) -> Dict:
    vessels = VesselRegistry(); records: List[Dict] = []
    steps = extract_steps(text)

    for step in steps:
        prep = detect_solution_prep(step)
        if prep:
            soln_name = f"{prep['solvent']} {prep['concentration']} {prep['concentration_units']} solution of {prep['solute']}"
            vid = vessels.ensure(soln_name)
            rec = {
                "action": "dispense",
                "vessel": vid,
                "solute": prep["solute"],
                "solvent": prep["solvent"],
                "concentration": prep["concentration"],
                "concentration_units": prep["concentration_units"],
                "concentration_kind": prep["concentration_kind"],
                "volume": prep["volume"],
                "volume_units": prep["volume_units"],
                "reagents": [prep["solute"], prep["solvent"]],
                "ops": ops_for_dispense(vid, prep["solute"], prep["solvent"], prep["volume"], prep["volume_units"]),
                "raw": step,
            }
            records.append(rec); continue

        add = detect_add_while_stirring(step)
        if add:
            src_key = re.sub(r"^\bthe\b\s+","",add["source_name"],flags=re.I).strip()
            dst_key = re.sub(r"^\bthe\b\s+","",add["target_name"],flags=re.I).strip()
            src_vid = vessels.lookup(src_key) or vessels.ensure(src_key)
            dst_vid = vessels.lookup(dst_key) or vessels.ensure(dst_key)
            rec = {
                "action": "add",
                "source_vessel": src_vid,
                "target_vessel": dst_vid,
                "reagents": [src_key],
                "with_stirring": add["with_stirring"],
                "volume": add["volume"],
                "volume_units": add["volume_units"],
                "ops": ops_for_add(src_vid, dst_vid, add["with_stirring"], "slow" if add["rate"]=="slow" else "normal", add["volume"], add["volume_units"]),
                "raw": step,
            }
            records.append(rec); continue

        sh = detect_stir_then_heat(step)
        if sh:
            target_vessel = vessels.primary_vessel or vessels.ensure("reaction mixture")
            for a in sh:
                if a["action"]=="stir":
                    records.append({"action":"stir","vessel":target_vessel,"reagents":[],"minutes":a["minutes"],
                                    "temperature_C":a["temperature_C"],"ops":ops_for_stir(target_vessel,a["minutes"],a["temperature_C"]),"raw":step})
                else:
                    records.append({"action":"heat_hold","vessel":target_vessel,"reagents":[],"minutes":a["minutes"],
                                    "temperature_C":a["temperature_C"],"ops":ops_for_heat(target_vessel,a["temperature_C"],a["minutes"]),"raw":step})
            continue

        pp = detect_postproc(step)
        if pp:
            target_vessel = vessels.primary_vessel or vessels.ensure("reaction mixture")
            records.append({"action":"postprocess","vessel":target_vessel,"reagents":[],"ops":ops_for_postproc(target_vessel,pp),"raw":step})
            continue

        target_vessel = vessels.primary_vessel or vessels.ensure("reaction mixture")
        records.append({"action":"process","vessel":target_vessel,"reagents":[],"ops":[],"raw":step})

    return {"vessel_registry": vessels.as_dict(), "devices": DEVICE_IDS, "defaults": DEFAULTS, "steps": records}

def explode_to_single_ops(doc: Dict) -> Dict:
    """Return a copy with steps replaced by single-op records (each op is one step)."""
    out = {k: v for k, v in doc.items() if k not in ("steps",)}
    single_steps: List[Dict] = []
    for step in doc["steps"]:
        base = {k: v for k, v in step.items() if k not in ("ops",)}
        ops = step.get("ops", [])
        if not ops:
            # still create a record to preserve ordering
            single_steps.append({**base, "op": None})
            continue
        for op in ops:
            rec = {**base, **{"op": op["op"]}}
            # merge op fields flat
            for k, v in op.items():
                if k == "op": continue
                rec[k] = v
            single_steps.append(rec)
    out["steps"] = single_steps
    return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python converter_robot_ready_v2.py <input.txt> <output.json> [--single-ops]")
        raise SystemExit(2)
    text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    doc = convert_text_to_robot_ops(text)
    if len(sys.argv) >= 4 and sys.argv[3] == "--single-ops":
        doc = explode_to_single_ops(doc)
    pathlib.Path(sys.argv[2]).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {len(doc['steps'])} steps.")
