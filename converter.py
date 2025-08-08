import re, json, pathlib
from typing import List, Dict, Optional

DEFAULTS = {
    "stir_rpm": 700,
    "centrifuge_rpm": 4000,
    "centrifuge_minutes": 10,
    "transfer_rate_slow": "slow",
    "room_temp_C": 25,
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
    mins = 0.0; found=False
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:hour|hr|hrs|h)\b", t, re.I):
        mins += float(m.group(1))*60; found=True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:minute|min|mins|m)\b", t, re.I):
        mins += float(m.group(1)); found=True
    return mins if found else None

# -------- Hardware parsing --------
def parse_hardware(markdown_text: str) -> List[Dict]:
    lines = markdown_text.splitlines()
    items = []
    in_hw = False
    for line in lines:
        if re.match(r"\s*1\.\s*\*\*Hardware\s*&\s*Glassware\*\*:", line, re.I):
            in_hw = True
            continue
        if in_hw:
            if line.strip().startswith("2.") or re.match(r"\s*2\.\s*\*\*", line):
                break
            if line.strip().startswith("- "):
                entry = strip_tags(line.strip()[2:])
                # Expand "Beakers (100 mL and 250 mL)" -> two items
                m = re.match(r"Beakers?\s*\((.+?)\)", entry, re.I)
                if m:
                    sizes = m.group(1)
                    parts = re.split(r"\s*(?:and|,)\s*", sizes)
                    for p in parts:
                        cap = p.strip()
                        items.append({"name": f"Beaker {cap}", "type": "beaker", "capacity": cap})
                else:
                    # Try to detect capacity in name
                    capm = re.search(r"(\d+)\s*(mL|L)\b", entry, re.I)
                    cap = capm.group(0) if capm else None
                    typ = "beaker" if "beaker" in entry.lower() else "hardware"
                    nm = entry if typ != "beaker" else (f"Beaker {cap}" if cap else "Beaker")
                    items.append({"name": nm, "type": typ, "capacity": cap})
    # give IDs
    out = []
    for i, it in enumerate(items, 1):
        it2 = dict(it); it2["id"] = f"H{i}"
        out.append(it2)
    return out

def _capacity_to_ml(cap: Optional[str]) -> Optional[float]:
    if not cap: return None
    m = re.match(r"(\d+(?:\.\d+)?)\s*(mL|L)\b", cap, re.I)
    if not m: return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    return val if unit=="ml" else val*1000.0

def _parse_vol_ml(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mL|ml|L|l)\b", text)
    if not m: return None
    val = float(m.group(1)); unit = m.group(2).lower()
    return val if unit=="ml" else val*1000.0

class VesselRegistry:
    def __init__(self, hardware: List[Dict]):
        self._vid_to_label: Dict[str,str] = {}
        self._label_to_vid: Dict[str,str] = {}
        self._vid_to_hid: Dict[str,str] = {}
        self._vessel_contents: Dict[str,str] = {}
        self._counter = 0
        self.primary_vessel: Optional[str] = None
        self.hardware = hardware

    def _new_vid(self) -> str:
        self._counter += 1
        return f"V{self._counter}"

    def _pick_beaker_for_volume(self, vol_ml: Optional[float]) -> Optional[Dict]:
        beakers = [h for h in self.hardware if h.get("type")=="beaker"]
        if not beakers: return None
        if vol_ml is None:
            # choose smallest beaker
            beakers_sorted = sorted(beakers, key=lambda h: (_capacity_to_ml(h.get("capacity")) or 1e9))
            return beakers_sorted[0]
        # choose smallest capacity >= vol_ml*1.5 safety factor
        target = vol_ml*1.5
        candidates = [(h, _capacity_to_ml(h.get("capacity")) or 1e12) for h in beakers]
        candidates = [c for c in candidates if c[1] >= target]
        if candidates:
            h, _ = sorted(candidates, key=lambda x: x[1])[0]
            return h
        # else pick largest
        h, _ = sorted([(h, _capacity_to_ml(h.get("capacity")) or 0) for h in beakers], key=lambda x: x[1], reverse=True)[0]
        return h

    def ensure_glassware(self, label: str, *, prefer_capacity_ml: Optional[float]=None, explicit_hardware_hint: Optional[str]=None) -> str:
        key = label.lower().strip()
        if key in self._label_to_vid:
            return self._label_to_vid[key]
        vid = self._new_vid()
        # pick hardware id
        hw_id = None
        if explicit_hardware_hint:
            for h in self.hardware:
                if explicit_hardware_hint.lower() in h["name"].lower():
                    hw_id = h["id"]; break
        if hw_id is None:
            chosen = self._pick_beaker_for_volume(prefer_capacity_ml)
            if chosen: hw_id = chosen["id"]
        self._vid_to_label[vid] = label
        self._label_to_vid[key] = vid
        if hw_id:
            self._vid_to_hid[vid] = hw_id
        if self.primary_vessel is None:
            self.primary_vessel = vid
        return vid

    def map_contents(self, vid: str, contents: str):
        self._vessel_contents[vid] = contents

    def vessel_hardware(self, vid: str) -> Optional[str]:
        return self._vid_to_hid.get(vid)

    def as_dict(self) -> Dict[str, str]:
        # glassware labels
        return dict(self._vid_to_label)

    def contents_dict(self) -> Dict[str, str]:
        return dict(self._vessel_contents)

_CONC_UNIT_RX = r"(?:M|m)"
def _clean_solvent_tail(solvent: str) -> str:
    solvent = strip_tags(solvent.strip().rstrip(",."))
    solvent = solvent.split(" in ")[0].strip()
    return solvent

def detect_solution_prep(line: str) -> Optional[Dict]:
    s = strip_tags(line.strip().rstrip("."))
    pats = [
        re.compile(
            rf"""prepare\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+solution\s+of\s+.+?\s+
                by\s+dissolving\s+(?:an\s+appropriate\s+amount\s+of\s+)?
                (?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s*(?:in\b|$)""",
            re.I|re.X),
        re.compile(
            rf"""prepare\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+(?P<xname>.+?)\s+solution\s+
                by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s*(?:in\b|$)""",
            re.I|re.X),
        re.compile(
            rf"""dissolv\w*\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s+
                to\s+(?:make|form|yield|obtain)\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+.+?\s+solution""",
            re.I|re.X),
    ]
    for rx in pats:
        m = rx.search(s)
        if m:
            conc = float(m.group(1))
            solute = m.groupdict().get("solute","").strip()
            vol = float(m.group("vol"))
            vunit = m.group("vunit")
            solvent = _clean_solvent_tail(m.group("solvent"))
            hint = None
            mh = re.search(r"in a\s+(\d+\s*(?:mL|L)\s+glass\s+beaker)", line, re.I)
            if mh: hint = mh.group(1)
            return {"action":"dispense","solute":solute,"solvent":solvent,"concentration":conc,"concentration_units":"M","volume":vol,"volume_units":vunit,"hardware_hint":hint}
    return None

def detect_add_while_stirring(line: str) -> Optional[Dict]:
    s = strip_tags(line.strip())
    m = re.search(r"\b(add|charge)\s+(?:the\s+)?(?P<src>.+?)\s+to\s+(?:the\s+)?(?P<dst>.+?)\s+(?:while\s+stirring|with\s+stirring|under\s+stirring)", s, re.I)
    if m:
        return {"action":"add","source_name":m.group("src").strip(),"target_name":m.group("dst").strip(),"with_stirring":True,"rate":"slow" if ("slow" in s.lower() or "dropwise" in s.lower()) else "normal"}
    m2 = re.search(r"\b(add|charge)\s+(?:the\s+)?(?P<src>.+?)\s+to\s+(?:the\s+)?(?P<dst>.+?)\b", s, re.I)
    if m2:
        return {"action":"add","source_name":m2.group("src").strip(),"target_name":m2.group("dst").strip(),"with_stirring":False,"rate":"normal"}
    return None

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

def ops_for_dispense(vessel: str, hardware_id: Optional[str], solute: str, solvent: str, volume_val: float, volume_unit: str) -> List[Dict]:
    ops = [{"op":"ensure_vessel","vessel":vessel,"hardware_id":hardware_id},
           {"op":"add_solute","vessel":vessel,"reagent":solute},
           {"op":"add_solvent","vessel":vessel,"solvent":solvent,"volume":volume_val,"volume_units":volume_unit}]
    return ops

def ops_for_add(src_v: str, dst_v: str, with_stir: bool, rate: str) -> List[Dict]:
    ops = []
    if with_stir:
        ops.append({"op":"move_to_stir_plate","vessel":dst_v,"stir_plate_id":"SP1"})
        ops.append({"op":"set_stir_rate","vessel":dst_v,"rpm":DEFAULTS["stir_rpm"]})
    ops.append({"op":"transfer","from":src_v,"to":dst_v,"rate":rate})
    return ops

def ops_for_stir(vessel: str, minutes: float) -> List[Dict]:
    return [{"op":"move_to_stir_plate","vessel":vessel,"stir_plate_id":"SP1"},
            {"op":"set_stir_rate","vessel":vessel,"rpm":DEFAULTS["stir_rpm"]},
            {"op":"wait","minutes":minutes}]

def ops_for_heat(vessel: str, temp_C: float, minutes: float) -> List[Dict]:
    return [{"op":"set_hotplate_temperature","hotplate_id":"HP1","temperature_C":temp_C},
            {"op":"wait","minutes":minutes}]

def ops_for_postproc(vessel: str, actions: List[Dict]) -> List[Dict]:
    ops = []
    for a in actions:
        if a["action"]=="cool_to_room":
            ops.append({"op":"set_hotplate_temperature","hotplate_id":"HP1","temperature_C":DEFAULTS["room_temp_C"]})
        elif a["action"]=="centrifuge":
            ops.append({"op":"transfer_to_centrifuge_tube","from":vessel,"to":f"{vessel}_tube"})
            ops.append({"op":"centrifuge","centrifuge_id":"CF1","rpm":a["rpm"],"minutes":a["minutes"]})
        elif a["action"]=="decant_supernatant":
            ops.append({"op":"decant_supernatant","tube":f"{vessel}_tube"})
        elif a["action"]=="add_wash_solvent":
            ops.append({"op":"add_wash_solvent","tube":f"{vessel}_tube","solvent":a["solvent"]})
        elif a["action"]=="resuspend":
            ops.append({"op":"resuspend","tube":f"{vessel}_tube"})
        elif a["action"]=="oven_dry":
            ops.append({"op":"move_to_oven","tube":f"{vessel}_tube","oven_id":"OV1"})
            ops.append({"op":"set_oven_temperature","oven_id":"OV1","temperature_C":a["temperature_C"]})
            ops.append({"op":"wait","minutes":a["minutes"]})
    return ops

def extract_steps(markdown_text: str) -> List[str]:
    lines = markdown_text.splitlines()
    in_proc = False; steps = []; buf = []
    for line in lines:
        if re.match(r"\s*3\.\s*\*\*Procedure\*\*:", line):
            in_proc = True; continue
        if in_proc:
            if re.match(r"\s*\d+\.\s", line):
                if buf: steps.append(" ".join(buf).strip()); buf = []
                buf.append(re.sub(r"^\s*\d+\.\s*", "", line).strip())
            else:
                if line.strip(): buf.append(line.strip())
    if buf: steps.append(" ".join(buf).strip())
    return [strip_tags(s) for s in steps if s.strip()]

def convert_text_to_robot_ops(text: str) -> Dict:
    hardware = parse_hardware(text)
    vessels = VesselRegistry(hardware)
    records: List[Dict] = []

    steps = extract_steps(text)

    for step in steps:
        prep = detect_solution_prep(step)
        if prep:
            # volume in mL
            vol_ml = prep["volume"] * (1.0 if prep["volume_units"].lower()=="ml" else 1000.0)
            explicit = prep.get("hardware_hint")
            label = explicit if explicit else "Beaker"
            vid = vessels.ensure_glassware(label, prefer_capacity_ml=vol_ml, explicit_hardware_hint=explicit)
            vessels.map_contents(vid, f"{prep['solvent']} {prep['concentration']} {prep['concentration_units']} solution of {prep['solute']}")
            hw_id = vessels.vessel_hardware(vid)
            records.append({
                "action":"dispense",
                "vessel": vid,
                "hardware_id": hw_id,
                "solute": prep["solute"],
                "solvent": prep["solvent"],
                "concentration": prep["concentration"],
                "concentration_units": prep["concentration_units"],
                "volume": prep["volume"],
                "volume_units": prep["volume_units"],
                "reagents": [prep["solute"], prep["solvent"]],
                "ops": ops_for_dispense(vid, hw_id, prep["solute"], prep["solvent"], prep["volume"], prep["volume_units"]),
                "raw": step,
            })
            continue

        add = detect_add_while_stirring(step)
        if add:
            src_key = re.sub(r"^\bthe\b\s+","",add["source_name"], flags=re.I).strip()
            dst_key = re.sub(r"^\bthe\b\s+","",add["target_name"], flags=re.I).strip()
            src_vid = vessels.ensure_glassware(src_key) if "beaker" in src_key.lower() else (vessels.primary_vessel or vessels.ensure_glassware("Beaker"))
            dst_vid = vessels.ensure_glassware(dst_key) if "beaker" in dst_key.lower() else (vessels.primary_vessel or vessels.ensure_glassware("Beaker"))
            records.append({
                "action": "add",
                "source_vessel": src_vid,
                "target_vessel": dst_vid,
                "reagents": [src_key],
                "with_stirring": add["with_stirring"],
                "ops": ops_for_add(src_vid, dst_vid, add["with_stirring"], "slow" if add["rate"]=="slow" else "normal"),
                "raw": step,
            })
            continue

        sh = detect_stir_then_heat(step)
        if sh:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            for a in sh:
                if a["action"]=="stir":
                    records.append({"action":"stir","vessel":target_vessel,"reagents":[],"minutes":a["minutes"],
                                    "temperature_C":a["temperature_C"],"ops":ops_for_stir(target_vessel,a["minutes"]), "raw": step})
                else:
                    records.append({"action":"heat_hold","vessel":target_vessel,"reagents":[],"minutes":a["minutes"],
                                    "temperature_C":a["temperature_C"],"ops":ops_for_heat(target_vessel,a["temperature_C"],a["minutes"]), "raw": step})
            continue

        pp = detect_postproc(step)
        if pp:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            records.append({"action":"postprocess","vessel":target_vessel,"reagents":[],"ops":ops_for_postproc(target_vessel,pp), "raw": step})
            continue

        target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
        records.append({"action":"process","vessel":target_vessel,"reagents":[],"ops":[], "raw": step})

    return {
        "hardware": hardware,
        "vessel_registry": vessels.as_dict(),   # glassware labels
        "vessel_contents": vessels.contents_dict(),
        "devices": DEVICE_IDS,
        "defaults": DEFAULTS,
        "steps": records,
    }