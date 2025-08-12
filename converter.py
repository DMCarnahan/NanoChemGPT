from __future__ import annotations

import re, json, pathlib, unicodedata
from typing import List, Dict, Optional, Any, Tuple

DEFAULTS = {
    "stir_rpm": 700,
    "centrifuge_rpm": 4000,
    "centrifuge_minutes": 10,
    "transfer_rate_slow": "slow",
    "room_temp_C": 25.0,
}

DEVICE_IDS = {
    "stir_plate_id": "SP1",
    "hotplate_id": "HP1",
    "centrifuge_id": "CF1",
    "oven_id": "OV1",
    "vacuum_pump_id": "VP1",
    "sonicator_id": "US1",
}

TAG_RX = re.compile(r"\s*\[(?:CTX|DB|PARSED|GEN|\d+)\]\s*$")

def strip_tags(s: str) -> str:
    return TAG_RX.sub("", s).strip()

def _clean_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.replace("° ", "°").replace("–", "-").replace("—", "-")

# -------- Units parsing --------
def find_temp_c(t: str) -> Optional[float]:
    s = _clean_unicode(t)
    if re.search(r"\breflux\b", s, re.I): return 100.0
    if re.search(r"\bboil(?:ing)?\b", s, re.I): return 100.0
    if re.search(r"\bice\s*bath\b", s, re.I): return 0.0
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CFK])\b", s, re.I)
    if not m:
        if re.search(r"\b(rt|room\s*temp(?:erature)?)\b", s, re.I): return DEFAULTS["room_temp_C"]
        return None
    val = float(m.group(1)); unit = m.group(2).upper()
    if unit == "C": return val
    if unit == "F": return (val - 32.0) * 5.0/9.0
    if unit == "K": return val - 273.15
    return None

def find_minutes(t: str) -> Optional[float]:
    s = _clean_unicode(t)
    mins = 0.0; found=False
    if re.search(r"\bover\s*night\b", s, re.I): return 12*60.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:second|sec|s)\b", s, re.I):
        mins += float(m.group(1))/60.0; found=True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:hour|hr|hrs|h)\b", s, re.I):
        mins += float(m.group(1))*60.0; found=True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:minute|min|mins|m)\b", s, re.I):
        mins += float(m.group(1)); found=True
    return mins if found else None

def _parse_vol_ml(text: str) -> Optional[float]:
    s = _clean_unicode(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(µ?u?L|mL|ml|L|l)\b", s)
    if not m: return None
    val = float(m.group(1)); unit = m.group(2).lower()
    if unit in ("µl","ul"): return val/1000.0
    return val if unit=="ml" else val*1000.0

def _parse_conc(text: str) -> Optional[Tuple[float,str]]:
    s = _clean_unicode(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m?M|%\s*w/?v|%\s*v/?v|%)\b", s, re.I)
    if not m: return None
    v = float(m.group(1)); u = m.group(2).replace(" ", "").lower()
    if u == "m": u = "M"
    if u == "mm": u = "mM"
    return v, u

# -------- Hardware parsing --------
def parse_hardware(markdown_text: str) -> List[Dict]:
    lines = markdown_text.splitlines()
    items = []
    in_hw = False
    for line in lines:
        if re.match(r"\s*1\.\s*\*\*Hardware\s*&\s*Glassware\*\*:", line, re.I):
            in_hw = True; continue
        if in_hw:
            if line.strip().startswith("2.") or re.match(r"\s*2\.\s*\*\*", line):
                break
            if line.strip().startswith("- "):
                entry = strip_tags(line.strip()[2:])
                m = re.match(r"(Beakers?|Flasks?)\s*\((.+?)\)", entry, re.I)
                if m:
                    base = "beaker" if "beaker" in m.group(1).lower() else "flask"
                    sizes = m.group(2)
                    parts = re.split(r"\s*(?:and|,)\s*", sizes)
                    for p in parts:
                        cap = p.strip()
                        items.append({"name": f"{m.group(1).split()[0].title()} {cap}", "type": base, "capacity": cap})
                else:
                    capm = re.search(r"(\d+)\s*(µ?u?L|mL|L)\b", entry, re.I)
                    cap = capm.group(0) if capm else None
                    typ = "beaker" if "beaker" in entry.lower() else ("flask" if "flask" in entry.lower() else "hardware")
                    nm = entry if typ == "hardware" else (f"{typ.title()} {cap}" if cap else typ.title())
                    items.append({"name": nm, "type": typ, "capacity": cap})
    out = []
    for i, it in enumerate(items, 1):
        it2 = dict(it); it2["id"] = f"H{i}"
        out.append(it2)
    return out

def _capacity_to_ml(cap: Optional[str]) -> Optional[float]:
    if not cap: return None
    m = re.match(r"(\d+(?:\.\d+)?)\s*(µ?u?L|mL|L)\b", cap, re.I)
    if not m: return None
    val = float(m.group(1)); unit = m.group(2).lower()
    if unit in ("µl","ul"): return val/1000.0
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

    def _pick_glass_for_volume(self, vol_ml: Optional[float], preferred: Optional[str]=None) -> Optional[Dict]:
        types = (preferred.lower(),) if preferred else ("beaker","flask")
        choices = [h for h in self.hardware if h.get("type") in types]
        if not choices: return None
        if vol_ml is None:
            return sorted(choices, key=lambda h: (_capacity_to_ml(h.get("capacity")) or 1e9))[0]
        target = vol_ml*1.5
        candidates = [(h, _capacity_to_ml(h.get("capacity")) or 1e12) for h in choices]
        candidates = [c for c in candidates if c[1] >= target]
        if candidates:
            h,_ = sorted(candidates, key=lambda x: x[1])[0]; return h
        h,_ = sorted([(h, _capacity_to_ml(h.get("capacity")) or 0) for h in choices], key=lambda x: x[1], reverse=True)[0]
        return h

    def ensure_glassware(self, label: str, *, prefer_capacity_ml: Optional[float]=None, explicit_hardware_hint: Optional[str]=None) -> str:
        key = label.lower().strip()
        if key in self._label_to_vid:
            return self._label_to_vid[key]
        vid = self._new_vid()
        hw_id = None
        preferred = None
        if explicit_hardware_hint:
            for h in self.hardware:
                if explicit_hardware_hint.lower() in h["name"].lower():
                    hw_id = h["id"]; preferred = h.get("type"); break
        if hw_id is None:
            chosen = self._pick_glass_for_volume(prefer_capacity_ml, preferred)
            if chosen: hw_id = chosen["id"]
        self._vid_to_label[vid] = label
        self._label_to_vid[key] = vid
        if hw_id: self._vid_to_hid[vid] = hw_id
        if self.primary_vessel is None: self.primary_vessel = vid
        return vid

    def map_contents(self, vid: str, contents: str): self._vessel_contents[vid] = contents
    def vessel_hardware(self, vid: str) -> Optional[str]: return self._vid_to_hid.get(vid)
    def as_dict(self) -> Dict[str, str]: return dict(self._vid_to_label)
    def contents_dict(self) -> Dict[str, str]: return dict(self._vessel_contents)

# -------- Pattern detectors --------
_CONC_UNIT_RX = r"(?:M|m|mM|%\s*w/?v|%\s*v/?v|%)"

def _clean_solvent_tail(solvent: str) -> str:
    solvent = strip_tags(solvent.strip().rstrip(",."))
    solvent = solvent.split(" in ")[0].strip()
    return solvent

def detect_solution_prep(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    pats = [
        re.compile(
            rf"""prepare\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+(?P<xname>.+?)\s+solution\s+
                by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s*(?:in\b|$)""",
            re.I|re.X),
        re.compile(
            rf"""dissolv\w*\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s+
                to\s+(?:make|form|yield|obtain)\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+.+?\s+solution""",
            re.I|re.X),
    ]
    for rx in pats:
        m = rx.search(s)
        if m:
            try:
                conc_val = float(m.group(1))
            except Exception:
                conc_val = None
            cu = m.group(2) if m.groups() else "M"
            if cu:
                cu = cu.replace(" ", "")
            solute = m.groupdict().get("solute","").strip()
            vol = float(m.group("vol"))
            vunit = m.group("vunit")
            solvent = _clean_solvent_tail(m.group("solvent"))
            hint = None
            mh = re.search(r"in a\s+(\d+\s*(?:µ?u?L|mL|L)\s+(?:glass\s+)?(?:beaker|flask))", s, re.I)
            if mh: hint = mh.group(1)
            return {"action":"dispense","solute":solute,"solvent":solvent,"concentration":conc_val,"concentration_units":cu,"volume":vol,"volume_units":vunit,"hardware_hint":hint}
    return None

def detect_add(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(r"\b(add|charge)\s+(?:the\s+)?(?P<src>.+?)\s+to\s+(?:the\s+)?(?P<dst>.+?)\b", s, re.I)
    if not m:
        return None
    rate = "slow" if re.search(r"\b(dropwise|slow)\b", s, re.I) else "normal"
    at_temp = find_temp_c(s)
    over_min = find_minutes(s)
    return {"action":"add","source_name":m.group("src").strip(),"target_name":m.group("dst").strip(),"rate":rate,"temperature_C":at_temp,"minutes":over_min}

def detect_stir(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if not re.search(r"\bstir", s, re.I): return None
    rpm = None
    mr = re.search(r"(\d{2,5})\s*rpm\b", s, re.I)
    if mr: rpm = int(mr.group(1))
    minutes = find_minutes(s) or 60.0
    temp = find_temp_c(s) or DEFAULTS["room_temp_C"]
    return {"action":"stir","rpm": rpm or DEFAULTS["stir_rpm"], "minutes": minutes, "temperature_C": temp}

def detect_heat(line: str) -> Optional[List[Dict]]:
    s = strip_tags(_clean_unicode(line.strip()))
    if not re.search(r"\bheat|\bmaintain|\bhold", s, re.I): return None
    temp = find_temp_c(s) or DEFAULTS["room_temp_C"]
    minutes = find_minutes(s) or 60.0
    return [{"action":"heat_to","temperature_C": temp}, {"action":"hold","minutes": minutes}]

def detect_cool(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\b(ice\s*bath|cool)\b", s, re.I):
        temp = 0.0 if "ice" in s.lower() else (find_temp_c(s) or DEFAULTS["room_temp_C"])
        return {"action":"cool_to","temperature_C": temp}
    return None

def detect_sonicate(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\bsonicat", s, re.I):
        return {"action":"sonicate","minutes": find_minutes(s) or 10.0}
    return None

def detect_filter(line: str) -> Optional[List[Dict]]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\b(vacuum\s*filter|filter\b)", s, re.I):
        ops = [{"action":"filter"}]
        if "vacuum" in s.lower(): ops.append({"action":"apply_vacuum"})
        return ops
    return None

def detect_wash_dry(line: str) -> Optional[List[Dict]]:
    s = strip_tags(_clean_unicode(line.strip()))
    ops = []
    if "wash" in s.lower():
        n = 1
        mw = re.search(r"(\d+)\s*[x×]\s*wash", s, re.I)
        if mw: n = int(mw.group(1))
        wash_solvent = "deionized water" if re.search(r"\b(di\s*water|deionized\s*water)\b", s, re.I) else "wash solvent"
        for _ in range(n):
            ops += [{"action":"add_wash_solvent","solvent":wash_solvent}, {"action":"resuspend"}, {"action":"centrifuge","rpm":DEFAULTS["centrifuge_rpm"],"minutes":DEFAULTS["centrifuge_minutes"]}, {"action":"decant_supernatant"}]
    if "dry" in s.lower() or "oven" in s.lower():
        temp = find_temp_c(s) or 60.0; minutes = find_minutes(s) or 120.0
        ops.append({"action":"oven_dry","temperature_C":temp,"minutes":minutes})
    return ops or None

# -------- Ops builders --------
def ops_for_dispense(vessel: str, hardware_id: Optional[str], solute: str, solvent: str, volume_val: float, volume_unit: str) -> List[Dict]:
    return [
        {"op":"ensure_vessel","vessel":vessel,"hardware_id":hardware_id},
        {"op":"add_solute","vessel":vessel,"reagent":solute},
        {"op":"add_solvent","vessel":vessel,"solvent":solvent,"volume":volume_val,"volume_units":volume_unit},
    ]

def ops_for_add(src_v: str, dst_v: str, rate: str, rpm: Optional[int]=None, temperature_C: Optional[float]=None, minutes: Optional[float]=None) -> List[Dict]:
    ops = [
        {"op":"move_to_stir_plate","vessel":dst_v,"stir_plate_id":DEVICE_IDS["stir_plate_id"]},
        {"op":"set_stir_rate","vessel":dst_v,"rpm": rpm or DEFAULTS["stir_rpm"]},
    ]
    if temperature_C is not None:
        ops.append({"op":"set_hotplate_temperature","hotplate_id":DEVICE_IDS["hotplate_id"],"temperature_C":temperature_C})
    ops.append({"op":"transfer","from":src_v,"to":dst_v,"rate":rate})
    if minutes:
        ops.append({"op":"wait","minutes":minutes})
    return ops

def ops_for_stir(vessel: str, minutes: float, rpm: int, temp_C: float) -> List[Dict]:
    return [
        {"op":"move_to_stir_plate","vessel":vessel,"stir_plate_id":DEVICE_IDS["stir_plate_id"]},
        {"op":"set_stir_rate","vessel":vessel,"rpm":rpm},
        {"op":"set_hotplate_temperature","hotplate_id":DEVICE_IDS["hotplate_id"],"temperature_C":temp_C},
        {"op":"wait","minutes":minutes},
    ]

def ops_for_heat(vessel: str, temp_C: float, minutes: float) -> List[Dict]:
    return [
        {"op":"set_hotplate_temperature","hotplate_id":DEVICE_IDS["hotplate_id"],"temperature_C":temp_C},
        {"op":"wait","minutes":minutes},
    ]

def ops_for_postproc(vessel: str, actions: List[Dict]) -> List[Dict]:
    ops = []
    for a in actions:
        if a["action"]=="cool_to":
            ops.append({"op":"set_hotplate_temperature","hotplate_id":DEVICE_IDS["hotplate_id"],"temperature_C":a["temperature_C"]})
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
        elif a["action"]=="filter":
            ops.append({"op":"setup_filtration"})
        elif a["action"]=="apply_vacuum":
            ops.append({"op":"start_vacuum","vacuum_pump_id":DEVICE_IDS["vacuum_pump_id"]})
        elif a["action"]=="sonicate":
            ops.append({"op":"sonicate","sonicator_id":DEVICE_IDS["sonicator_id"],"minutes":a.get("minutes",10.0)})
    return ops

# -------- Step extraction --------
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

# -------- Main converter --------
def convert_text_to_robot_ops(text: str) -> Dict:
    hardware = parse_hardware(text)
    vessels = VesselRegistry(hardware)
    records: List[Dict] = []

    steps = extract_steps(text)

    for step in steps:
        # Solution preparation
        prep = detect_solution_prep(step)
        if prep:
            vol_ml = prep["volume"] * (0.001 if prep["volume_units"].lower() in ("µl","ul") else (1.0 if prep["volume_units"].lower()=="ml" else 1000.0))
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

        # Additions (with optional temp/rate/time)
        add = detect_add(step)
        if add:
            src_key = re.sub(r"^\bthe\b\s+","",add["source_name"], flags=re.I).strip()
            dst_key = re.sub(r"^\bthe\b\s+","",add["target_name"], flags=re.I).strip()
            src_vid = vessels.ensure_glassware(src_key) if "beaker" in src_key.lower() or "flask" in src_key.lower() else (vessels.primary_vessel or vessels.ensure_glassware("Beaker"))
            dst_vid = vessels.ensure_glassware(dst_key) if "beaker" in dst_key.lower() or "flask" in dst_key.lower() else (vessels.primary_vessel or vessels.ensure_glassware("Beaker"))
            records.append({
                "action": "add",
                "source_vessel": src_vid,
                "target_vessel": dst_vid,
                "reagents": [src_key],
                "with_stirring": True,
                "rate": add["rate"],
                "temperature_C": add.get("temperature_C"),
                "minutes": add.get("minutes"),
                "ops": ops_for_add(src_vid, dst_vid, add["rate"], temperature_C=add.get("temperature_C"), minutes=add.get("minutes")),
                "raw": step,
            })
            continue

        # Stirring
        st = detect_stir(step)
        if st:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            records.append({
                "action":"stir","vessel":target_vessel,"reagents":[],
                "minutes":st["minutes"], "temperature_C":st["temperature_C"], "rpm":st["rpm"],
                "ops":ops_for_stir(target_vessel, st["minutes"], st["rpm"], st["temperature_C"]), "raw": step
            })
            continue

        # Heating
        ht = detect_heat(step)
        if ht:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            temp = ht[0]["temperature_C"]; minutes = ht[1]["minutes"]
            records.append({
                "action":"heat_hold","vessel":target_vessel,"reagents":[],
                "minutes": minutes, "temperature_C": temp,
                "ops": ops_for_heat(target_vessel, temp, minutes), "raw": step
            })
            continue

        # Cooling
        cl = detect_cool(step)
        if cl:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            records.append({
                "action":"cool_to","vessel":target_vessel,"reagents":[],
                "temperature_C": cl["temperature_C"],
                "ops": [{"op":"set_hotplate_temperature","hotplate_id":DEVICE_IDS["hotplate_id"],"temperature_C":cl["temperature_C"]}],
                "raw": step
            })
            continue

        # Sonication
        so = detect_sonicate(step)
        if so:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            records.append({
                "action":"sonicate","vessel":target_vessel,"reagents":[],
                "minutes": so["minutes"],
                "ops": [{"op":"sonicate","sonicator_id":DEVICE_IDS["sonicator_id"],"minutes":so["minutes"]}],
                "raw": step
            })
            continue

        # Filtration / washing / drying
        filt = detect_filter(step)
        if filt:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            records.append({"action":"postprocess","vessel":target_vessel,"reagents":[],"ops":ops_for_postproc(target_vessel,filt), "raw": step})
            continue

        wd = detect_wash_dry(step)
        if wd:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            records.append({"action":"postprocess","vessel":target_vessel,"reagents":[],"ops":ops_for_postproc(target_vessel,wd), "raw": step})
            continue

        # Fallback generic process node
        target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
        records.append({"action":"process","vessel":target_vessel,"reagents":[],"ops":[], "raw": step})

    return {
        "hardware": hardware,
        "vessel_registry": vessels.as_dict(),
        "vessel_contents": vessels.contents_dict(),
        "devices": DEVICE_IDS,
        "defaults": DEFAULTS,
        "steps": records,
    }

# -------- Validation helpers (unchanged API) --------
def validate_step(text: str) -> Dict[str, Any]:
    if not isinstance(text, str): raise ValueError("input must be a string")
    raw = text.strip()
    if not raw: raise ValueError("input text is empty")
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict): return obj
        raise ValueError("JSON input must be an object")
    except json.JSONDecodeError:
        pass
    data: Dict[str, Any] = {}
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip(): continue
        key, value = line.split(":", 1)
        key = key.strip(); value = value.strip()
        if not key: raise ValueError(f"line {lineno}: key is empty")
        data[key] = value
    if not data: raise ValueError("no key:value pairs found")
    return data

def validate_file(path: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists(): raise ValueError(f"file '{path}' does not exist")
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip(): continue
            try:
                item = validate_step(line)
            except ValueError as ve:
                raise ValueError(f"{p.name}:{lineno}: {ve}") from None
            items.append(item)
    return items

# -------- CLI --------
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Convert a TXT/MD protocol to robot JSON ops")
    ap.add_argument("path", help="Input file path")
    ap.add_argument("-o", "--out", default="-", help="Output JSON path (default stdout)")
    args = ap.parse_args()
    txt = pathlib.Path(args.path).read_text(encoding="utf-8", errors="ignore")
    obj = convert_text_to_robot_ops(txt)
    js = json.dumps(obj, indent=2, ensure_ascii=False)
    if args.out == "-":
        print(js)
    else:
        pathlib.Path(args.out).write_text(js, encoding="utf-8")
        print(f"Wrote {args.out}")
