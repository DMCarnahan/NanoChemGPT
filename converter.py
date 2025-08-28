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

FENCE_START_RX = re.compile(r"^\s*```")                    # start of any fenced block
NON_PROC_HEAD_RX = re.compile(
    r"^\s*#{1,6}\s*(references?|sources?|bibliography|rationale|reasoning|notes|discussion|supplementary|appendix|acknowledge?ments?)\b",
    re.I
)
INLINE_TAG_RX = re.compile(r"\s*\[(?:CTX|DB|PARSED|GEN|\d+)\]\s*", re.I)

# quantities like: 0.5 mmol | 58 mg | 10 mL | 1–2 mmol (range ok)
_AMOUNT_UNIT = r"(?:~?\d+(?:[.\u2013\u2014-]\d+)?\s*(?:µ?u?L|mL|ml|L|l|mg|g|µg|ug|mol|mmol|µmol|umol)\b)"

# split boundary: comma / + / "and" / "along with" / "together with"
# only split if the next token looks like a fresh quantity+unit
SPLIT_BOUNDARY_RX = re.compile(
    rf"\s*(?:,|\+|\band\b|\balong with\b|\btogether with\b)\s*(?=(?:{_AMOUNT_UNIT}))",
    re.I,
)

# Canonicalize unit spellings
_UNIT_CANON = {
    "l": "L", "L": "L",
    "ml": "mL", "mL": "mL",
    "ul": "µL", "uL": "µL", "µl": "µL", "µL": "µL",
    "g": "g", "mg": "mg", "µg": "µg", "ug": "µg",
    "mol": "mol", "mmol": "mmol", "µmol": "µmol", "umol": "µmol",
    "m": "M", "M": "M", "mm": "mM", "mM": "mM", "µM": "µM", "uM": "µM",
    "wt%": "wt%", "vol%": "vol%"
}

# Amount (mass/volume/moles) like: 98 mg | 10 mL | 0.5 mmol | 1–2 mmol
_AMOUNT_RE = r"(?P<approx>[~≈])?\s*(?P<val>\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?)\s*(?P<unit>µ?u?L|mL|ml|L|l|mg|g|µg|ug|mol|mmol|µmol|umol)\b"

# Secondary amount in parentheses: (98 mg), (10 mL)
_PAREN_AMOUNT_RX = re.compile(r"\(\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>µ?u?L|mL|ml|L|l|mg|g|µg|ug|mol|mmol|µmol|umol)\s*\)")

# Concentration form: 0.1 M HAuCl4 [in water]
_CONC_RX = re.compile(
    r"(?P<approx>[~≈])?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|µM|uM)\s+(?P<name>[^(),;]+?)(?:\s+in\s+(?P<solvent>[^(),;]+))?\b",
    re.I
)

# Leading amount form: 98 mg PVP | 0.5 mmol of copper(II) acetate ...
_LEAD_AMT_RX = re.compile(
    rf"{_AMOUNT_RE}" + r"\s*(?:of\s+)?(?P<name>.+?)\s*(?P<paren>\([^)]*\))?\s*$",
    re.I
)

# Loose "about/approximately" flags anywhere
_APPROX_WORD_RX = re.compile(r"\b(about|approximately|approx\.)\b", re.I)

def _canon_unit(u: str) -> str:
    return _UNIT_CANON.get((u or "").strip(), (u or "").strip())

def _to_float_range(s: str) -> tuple[float, float] | tuple[float, None]:
    """Parse '1–2' or '1-2' into (1.0, 2.0); else single -> (x, None)."""
    s = s.replace("–", "-")
    if "-" in s and not s.startswith("-"):
        a, b = s.split("-", 1)
        try:
            return (float(a), float(b))
        except Exception:
            pass
    try:
        return (float(s), None)
    except Exception:
        return (0.0, None)

def strip_tags(s: str) -> str:
    s = _clean_unicode(s)
    s = re.sub(r"`{3,}.*$", "", s)
    s = INLINE_TAG_RX.sub(" ", s)
    s = re.sub(r"</?[^>]+>", "", s)   # <-- new
    s = s.replace("**","").replace("__","")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def split_reagent_phrases(text: str) -> list[str]:
    """
    Split a multi-reagent phrase into separate items:
    e.g., '0.5 mmol Cu(OAc)2 (98 mg) and 0.5 mmol PVP (58 mg)'
      -> ['0.5 mmol Cu(OAc)2 (98 mg)', '0.5 mmol PVP (58 mg)']
    Only splits where a new quantity+unit begins; avoids over-splitting names.
    """
    s = (text or "").strip()
    if not s:
        return []
    parts = re.split(SPLIT_BOUNDARY_RX, s)
    out = []
    for p in parts:
        p = p.strip().strip(",").strip()
        if p:
            out.append(p)
    return out

def parse_reagent_phrase_to_struct(s: str) -> dict:
    """
    Parse a single reagent phrase into a structured dict.
    Supports:
      - '0.5 mmol copper(II) acetate monohydrate (98 mg)'
      - '98 mg PVP'
      - '10 mL ethylene glycol'
      - '0.1 M HAuCl4 in water'
    Returns a dict with keys:
      name, amount, amount_unit, amount_range, alt_amount, alt_unit,
      concentration, conc_unit, solvent, approx, original
    """
    original = s
    s = strip_tags(_clean_unicode((s or "").strip()))
    approx = bool(_APPROX_WORD_RX.search(s))

    # 1) Try concentration pattern first
    m = _CONC_RX.search(s)
    if m:
        val = float(m.group("val"))
        unit = _canon_unit(m.group("unit"))
        name = m.group("name").strip()
        solvent = (m.group("solvent") or "").strip() or None
        approx = approx or bool(m.group("approx"))
        return {
            "name": name,
            "amount": None,
            "amount_unit": None,
            "amount_range": None,
            "alt_amount": None,
            "alt_unit": None,
            "concentration": val,
            "conc_unit": unit,
            "solvent": solvent,
            "approx": approx,
            "original": original
        }

    # 2) Try leading amount pattern
    m = _LEAD_AMT_RX.match(s)
    if m:
        rng = _to_float_range(m.group("val"))
        amount = rng[0]
        amount_range = None
        if rng[1] is not None:
            amount_range = [rng[0], rng[1]]

        unit = _canon_unit(m.group("unit"))
        name = (m.group("name") or "").strip().strip(",;")
        approx = approx or bool(m.group("approx"))

        # Optional secondary amount in parentheses
        alt_amount = None
        alt_unit = None
        par = m.group("paren") or ""
        pm = _PAREN_AMOUNT_RX.search(par)
        if pm:
            alt_amount = float(pm.group("val"))
            alt_unit = _canon_unit(pm.group("unit"))

        return {
            "name": name,
            "amount": amount,
            "amount_unit": unit,
            "amount_range": amount_range,
            "alt_amount": alt_amount,
            "alt_unit": alt_unit,
            "concentration": None,
            "conc_unit": None,
            "solvent": None,
            "approx": approx,
            "original": original
        }

    # 3) Fallback: just return name
    return {
        "name": s,
        "amount": None,
        "amount_unit": None,
        "amount_range": None,
        "alt_amount": None,
        "alt_unit": None,
        "concentration": None,
        "conc_unit": None,
        "solvent": None,
        "approx": approx,
        "original": original
    }

def _clean_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.replace("° ", "°").replace("–", "-").replace("—", "-")

def _normalize_reagents_inplace(record: dict) -> None:
    # reagents: flatten and split strings
    reag = record.get("reagents", [])
    flat: list[str] = []
    for item in (reag if isinstance(reag, list) else [reag]):
        if isinstance(item, str):
            flat.extend(split_reagent_phrases(item))
        elif item:
            # preserve dicts/structured entries 
            flat.append(item)
    record["reagents"] = flat

    solute_str = record.get("solute", "")
    if isinstance(solute_str, str) and solute_str.strip():
        record["solutes"] = split_reagent_phrases(solute_str)

def _add_structured_reagents_inplace(record: dict) -> None:
    """
    Populate record['reagents_structured'] from record['reagents'] (strings).
    Also populate record['solutes_structured'] from record['solutes'] if present.
    """
    reag = record.get("reagents", []) or []
    if not isinstance(reag, list):
        reag = [reag]
    record["reagents_structured"] = [parse_reagent_phrase_to_struct(x) for x in reag if isinstance(x, str) and x.strip()]

    solutes = record.get("solutes", []) or []
    if isinstance(solutes, list) and solutes:
        record["solutes_structured"] = [parse_reagent_phrase_to_struct(x) for x in solutes if isinstance(x, str) and x.strip()]

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
                    hw_id = h["id"]
                    preferred = h.get("type")
                    break
        if hw_id is None:
            chosen = self._pick_glass_for_volume(prefer_capacity_ml, preferred)
            if chosen:
                hw_id = chosen["id"]
        self._vid_to_label[vid] = label
        self._label_to_vid[key] = vid
        if hw_id:
            self._vid_to_hid[vid] = hw_id
        if self.primary_vessel is None:
            self.primary_vessel = vid
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
        re.compile(
            r"dissolv\w*\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+of\s+(?P<solvent>.+?)(?:\.|$)",
            re.I
        ),
    ]
    for rx in pats:
        m = rx.search(s)
        if m:
            solute = m.groupdict().get("solute","").strip()
            vol = float(m.group("vol"))
            vunit = m.group("vunit")
            solvent = _clean_solvent_tail(m.group("solvent"))
            hint = None
            mh = re.search(r"in a\s+(\d+\s*(?:µ?u?L|mL|L)\s+(?:glass\s+)?(?:beaker|flask))", s, re.I)
            if mh: hint = mh.group(1)
            conc_val, conc_unit = None, None
            conc_match = re.search(r"(\d+(?:\.\d+)?)\s*(M|mM|%)\s+solution", s)
            if conc_match:
                conc_val = float(conc_match.group(1))
                conc_unit = conc_match.group(2)
            return {
                "action":"dispense",
                "solute":solute,
                "solvent":solvent,
                "concentration":conc_val,
                "concentration_units":conc_unit,
                "volume":vol,
                "volume_units":vunit,
                "hardware_hint":hint
            }
    return None

def detect_add_solvent(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(
        r"\badd\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s+to\s+(?:the\s+)?(?:solution|mixture|suspension|dispersion)\b",
        s, re.I
    )
    if not m:
        return None
    return {
        "action": "add_solvent",
        "volume": float(m.group("vol")),
        "volume_units": m.group("vunit"),
        "solvent": m.group("solvent").strip()
    }

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
    if not re.search(r"\b(heat|maintain|hold)\b", s, re.I): return None
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

def detect_resuspend(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\bresuspend\b", s, re.I):
        return {"action": "resuspend"}
    return None

def detect_collect(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\bcollect\b", s, re.I):
        return {"action": "collect"}
    return None

def detect_discard(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\bdiscard\b", s, re.I):
        return {"action": "discard"}
    return None

def detect_transfer(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(r"\btransfer\b.*\bto\b\s+(?P<target>.+)", s, re.I)
    if m:
        return {"action": "transfer", "target": m.group("target").strip()}
    return None

def detect_weigh(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(r"\bweigh\s+(?P<amount>[\d\.]+)\s*(?P<unit>mg|g|µg|kg)\s+of\s+(?P<reagent>.+?)(?:\.|$)", s, re.I)
    if m:
        return {
            "action": "weigh",
            "reagent": m.group("reagent").strip(),
            "amount": float(m.group("amount")),
            "unit": m.group("unit")
        }
    return None

def detect_transfer_explicit(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(r"\btransfer\s+(?:it|the\s+mixture|solution|precipitate)?\s*(?:into|to)\s+(?P<target>.+?)(?:\.|$)", s, re.I)
    if m:
        return {
            "action": "transfer",
            "target": m.group("target").strip()
        }
    return None



def detect_dissolve(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(
        r"\bdissolv\w*\s+(?P<amount>[\d\.]+)\s*(?P<unit>mg|g|µg|kg)\s+of\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+of\s+(?P<solvent>[^.;,]+)",
        s, re.I
    )
    if not m:
        return None

    solute = m.group("solute").strip()
    solvent_captured = _clean_solvent_tail(m.group("solvent").strip())
    vol1 = float(m.group("vol"))
    vunit1 = m.group("vunit")

    extras = []
    inline = solvent_captured
    while True:
        exm = re.search(r"(.*?)(?:,\s*)?(?:and\s+)([\d\.]+)\s*(µ?u?L|mL|ml|l|L)\s+of\s+([^,;]+)$", inline, re.I)
        if not exm:
            break
        base = exm.group(1).strip()
        vol2 = float(exm.group(2)); vunit2 = exm.group(3)
        solv2 = _clean_solvent_tail(exm.group(4).strip())
        extras.insert(0, {"name": solv2, "volume": vol2, "volume_units": vunit2})
        inline = base
    solvent1 = inline

    hint = None
    mh = re.search(r"in\s+(?:a|the)\s+(\d+\s*(?:µ?u?L|mL|L)\s+(?:glass\s+)?(?:beaker|flask|round-?bottom\s+flask))", s, re.I)
    if mh:
        hint = mh.group(1)

    result = {
        "action": "dissolve",
        "solute": solute,
        "amount": float(m.group("amount")),
        "unit": m.group("unit"),
        "solvent": solvent1 if not extras else solvent1 + " + " + " + ".join(e["name"] for e in extras),
        "volume": vol1,
        "volume_units": vunit1,
        "hardware_hint": hint,
    }
    if extras:
        result["solvents"] = [{"name": solvent1, "volume": vol1, "volume_units": vunit1}] + extras
    return result

    solute = m.group("solute").strip()
    solvent1 = _clean_solvent_tail(m.group("solvent").strip())
    # handle inline extra solvents like "ethylene glycol and 5 mL of water"
    extras = []
    inline = solvent1
    # repeatedly peel off trailing ", and 5 mL of X" or "and 5 mL of X"
    while True:
        exm = re.search(r"(.*?)(?:,\s*)?(?:and\s+)([\d\.]+)\s*(µ?u?L|mL|ml|l|L)\s+of\s+([^,;]+)$", inline, re.I)
        if not exm:
            break
        base = exm.group(1).strip()
        vol2 = float(exm.group(2)); vunit2 = exm.group(3)
        solv2 = _clean_solvent_tail(exm.group(4).strip())
        extras.insert(0, {"name": solv2, "volume": vol2, "volume_units": vunit2})
        inline = base
    solvent1 = inline

    vol1 = float(m.group("vol"))
    vunit1 = m.group("vunit")

    hint = None
    mh = re.search(r"in\s+(?:a|the)\s+(\d+\s*(?:µ?u?L|mL|L)\s+(?:glass\s+)?(?:beaker|flask|round-?bottom\s+flask))", s, re.I)
    if mh:
        hint = mh.group(1)

    result = {
        "action": "dissolve",
        "solute": solute,
        "amount": float(m.group("amount")),
        "unit": m.group("unit"),
        "solvent": solvent1 if not extras else solvent1 + " + " + " + ".join(e["name"] for e in extras),
        "volume": vol1,
        "volume_units": vunit1,
        "hardware_hint": hint,
    }
    if extras:
        result["solvents"] = [{"name": solvent1, "volume": vol1, "volume_units": vunit1}] + extras
    return result


def detect_filter_isolate(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\b(isolate|collect|obtain)\s+(?:the\s+)?(precipitate|solid|product)", s, re.I):
        return {"action": "isolate"}
    return None

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
        ops.append({"op":"set","device":DEVICE_IDS["hotplate_id"],"param":"temperature_C","value":temperature_C})
    ops.append({"op":"transfer","from":src_v,"to":dst_v,"rate":rate})
    if minutes:
        ops.append({"op":"wait","minutes":minutes})
    return ops

def ops_for_stir(vessel: str, minutes: float, rpm: int, temp_C: float) -> List[Dict]:
    return [
        {"op":"move_to_stir_plate","vessel":vessel,"stir_plate_id":DEVICE_IDS["stir_plate_id"]},
        {"op":"set_stir_rate","vessel":vessel,"rpm":rpm},
        {"op":"set","device":DEVICE_IDS["hotplate_id"],"param":"temperature_C","value":temp_C},
        {"op":"wait","minutes":minutes},
    ]

def ops_for_heat(vessel: str, temp_C: float, minutes: float) -> List[Dict]:
    return [
        {"op":"set","device":DEVICE_IDS["hotplate_id"],"param":"temperature_C","value":temp_C},
        {"op":"wait","minutes":minutes},
    ]

def ops_for_postproc(vessel: str, actions: List[Dict]) -> List[Dict]:
    ops = []
    for a in actions:
        if a["action"]=="cool_to":
            ops.append({"op":"set","device":DEVICE_IDS["hotplate_id"],"param":"temperature_C","value":a["temperature_C"]})
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
            ops.append({"op":"set","device":DEVICE_IDS["oven_id"],"param":"temperature_C","value":a["temperature_C"]})
            ops.append({"op":"wait","minutes":a["minutes"]})
        elif a["action"]=="filter":
            ops.append({"op":"setup_filtration"})
        elif a["action"]=="apply_vacuum":
            ops.append({"op":"start","device":DEVICE_IDS["vacuum_pump_id"]})
        elif a["action"]=="sonicate":
            ops.append({"op":"sonicate","sonicator_id":DEVICE_IDS["sonicator_id"],"minutes":a.get("minutes",10.0)})
    return ops

# -------- Step extraction --------
def extract_steps(markdown_text: str) -> List[str]:
    lines = markdown_text.splitlines()
    in_proc = False; steps = []; buf = []
    for line in lines:
        if re.match(r"^\s*\d+\.\s*\*\*Procedure\*\*:", line):
            in_proc = True
            continue
        if in_proc:
            # hard stops to avoid pulling rich-text sections
            if FENCE_START_RX.match(line) or NON_PROC_HEAD_RX.match(line):
                break

            if re.match(r"\s*\d+\.\s", line):
                if buf:
                    steps.append(" ".join(buf).strip()); buf = []
                buf.append(re.sub(r"^\s*\d+\.\s*", "", line).strip())
            else:
                # ignore empty or fence continuation lines if any slipped in
                if line.strip() and not line.strip().startswith("```"):
                    buf.append(line.strip())
    if buf:
        steps.append(" ".join(buf).strip())
    return [strip_tags(s) for s in steps if s.strip()]

# -------- Main converter --------

# -------- Micro-action expansion --------
def _label_for_vessel(vid: str, vessels: 'VesselRegistry', hardware: list[dict]) -> str:
    if not vid:
        vid = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
    label = vessels.as_dict().get(vid) or vid
    hid = vessels.vessel_hardware(vid)
    if hid:
        hw = next((h for h in hardware if h.get("id")==hid), None)
        if hw and hw.get("name"):
            return f"{hw['name']} ({label})"
    return label

def _micro_for_op(op: dict, vessels: 'VesselRegistry', hardware: list[dict]) -> list[dict]:
    m = []
    typ = op.get("op")
    if typ == "ensure_vessel":
        v = _label_for_vessel(op.get("vessel",""), vessels, hardware)
        m += [{"verb":"pick_up","object":v,"from":"bench"},
              {"verb":"place","object":v,"to":"bench"}]
        return m
    if typ == "move_to_stir_plate":
        v = _label_for_vessel(op.get("vessel",""), vessels, hardware)
        m += [{"verb":"pick_up","object":v,"from":"bench"},
              {"verb":"place","object":v,"to":"stir_plate"}]
        return m
    if typ == "set_stir_rate":
        m += [{"verb":"set","device": op.get("stir_plate_id") or "stir_plate",
            "param":"rpm","value":op.get("rpm")}]
        return m

    # Generic primitive ops -----------------------------------------------
    if typ == "set":
        dev = op.get("device") or op.get("hotplate_id") or op.get("oven_id") or "device"
        m += [{"verb":"set","device": dev, "param": op.get("param"), "value": op.get("value")}]
        return m
    if typ == "start":
        dev = op.get("device") or "device"
        m += [{"verb":"start","device": dev}]
        return m
    if typ == "stop":
        dev = op.get("device") or "device"
        m += [{"verb":"stop","device": dev}]
        return m
    if typ == "pick_up":
        v = _label_for_vessel(op.get("vessel",""), vessels, hardware)
        m += [{"verb":"pick_up","object":v,"from":"bench"}]
        return m
    if typ == "place":
        v = _label_for_vessel(op.get("vessel",""), vessels, hardware)
        m += [{"verb":"place","object":v,"to":"bench"}]
        return m
    if typ == "transfer":
        src = _label_for_vessel(op.get("from", ""), vessels, hardware) if op.get("from") else "source"
        dst = _label_for_vessel(op.get("to", ""),   vessels, hardware) if op.get("to")   else "target"
        rate = op.get("rate") or "normal"
        m += [
            {"verb":"pick_up","object":src,"from":"bench"},
            {"verb":"pour","from":src,"to":dst,"rate":rate},
            {"verb":"place","object":src,"to":"bench"},
        ]
        return m
    if typ == "wait":
        m += [{"verb":"wait","minutes": op.get("minutes")}]
        return m
    if typ == "filter":
        v = _label_for_vessel(op.get("vessel",""), vessels, hardware)
        m += [{"verb":"place","object":"filtration setup","to":"bench"},
              {"verb":"pick_up","object":v,"from":"bench"},
              {"verb":"pour","from":v,"to":"filtration setup"},
              {"verb":"place","object":v,"to":"bench"}]
        return m
    if typ == "start_vacuum":
        m += [{"verb":"set","device": op.get("vacuum_pump_id") or "vacuum_pump","param":"power","value":"on"},
              {"verb":"start","device": op.get("vacuum_pump_id") or "vacuum_pump"}]
        return m
    if typ == "decant_supernatant":
        tube = op.get("tube") or "tube"
        m += [{"verb":"pick_up","object":tube,"from":"rack"},
              {"verb":"pour","from":tube,"to":"waste"},
              {"verb":"place","object":tube,"to":"rack"}]
        return m

    if typ == "add_wash_solvent":
        tube = op.get("tube") or "tube"
        src = op.get("solvent") or "wash solvent"
        m += [{"verb":"pick_up","object":src,"from":"bench"},
              {"verb":"pour","from":src,"to":tube},
              {"verb":"place","object":src,"to":"bench"}]
        return m

    if typ == "resuspend":
        tube = op.get("tube") or "tube"
        m += [{"verb":"pick_up","object":tube,"from":"rack"},
              {"verb":"place","object":tube,"to":"vortex"},
              {"verb":"wait","minutes":1},
              {"verb":"place","object":tube,"to":"rack"}]
        return m

    if typ == "centrifuge":
        tube = "tube"
        m += [{"verb":"pick_up","object":tube,"from":"rack"},
              {"verb":"place","object":tube,"to":"centrifuge"},
              {"verb":"set","device": op.get("centrifuge_id") or "centrifuge","param":"rpm","value": op.get("rpm")},
              {"verb":"set","device": op.get("centrifuge_id") or "centrifuge","param":"minutes","value": op.get("minutes")},
              {"verb":"start","device": op.get("centrifuge_id") or "centrifuge"},
              {"verb":"wait","minutes": op.get("minutes")},
              {"verb":"stop","device": op.get("centrifuge_id") or "centrifuge"},
              {"verb":"place","object":tube,"to":"rack"}]
        return m

    if typ == "sonicate":
        tube = "tube"
        m += [{"verb":"pick_up","object":tube,"from":"rack"},
              {"verb":"place","object":tube,"to":"sonicator"},
              {"verb":"set","device": op.get("sonicator_id") or "sonicator","param":"minutes","value": op.get("minutes")},
              {"verb":"start","device": op.get("sonicator_id") or "sonicator"},
              {"verb":"wait","minutes": op.get("minutes")},
              {"verb":"stop","device": op.get("sonicator_id") or "sonicator"},
              {"verb":"place","object":tube,"to":"rack"}]
        return m

    if typ == "move_to_oven":
        m += [{"verb":"pick_up","object":op.get("tube") or "tube","from":"rack"},
              {"verb":"place","object":op.get("tube") or "tube","to":"oven"}]
        return m

    if typ == "set_oven_temperature":
        m += [{"verb":"set","device": op.get("oven_id") or "oven","param":"temperature_C","value": op.get("temperature_C")}]
        return m

    if typ == "stir":
        v = _label_for_vessel(op.get("vessel",""), vessels, hardware)
        m += [{"verb":"place","object":v,"to":"stir_plate"},
              {"verb":"set","device":"stir_plate","param":"rpm","value": op.get("rpm")},
              {"verb":"wait","minutes": op.get("minutes")}]
        return m

    return m

def expand_ops_to_micro(ops: list[dict], vessels: 'VesselRegistry', hardware: list[dict]) -> list[dict]:
    out = []
    for op in ops or []:
        out.extend(_micro_for_op(op, vessels, hardware))
    return out

def convert_text_to_robot_ops(text: str) -> Dict:
    hardware = parse_hardware(text)
    vessels = VesselRegistry(hardware)
    records: List[Dict] = []

    steps = extract_steps(text)

    for step in steps:
        # Weighing
        weigh = detect_weigh(step)
        if weigh:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "weigh",
                "vessel": target_vessel,
                "reagent": weigh["reagent"],
                "amount": weigh["amount"],
                "unit": weigh["unit"],
                "ops": [{"op": "weigh", "reagent": weigh["reagent"], "amount": weigh["amount"], "unit": weigh["unit"]}],
                "raw": step,
                "reagents": [weigh["reagent"]]
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Transfer (explicit)
        transfer_exp = detect_transfer_explicit(step)
        if transfer_exp:
            target_vessel = vessels.ensure_glassware(transfer_exp["target"])
            record = {
                "action": "transfer",
                "vessel": target_vessel,
                "ops": [{"op": "transfer", "to": transfer_exp["target"], "tube": f"{target_vessel}_tube"}],
                "raw": step,
                "reagents": []
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Dissolve
        dissolve = detect_dissolve(step)
        if dissolve:
            target_vessel = vessels.ensure_glassware(f"{dissolve['solute']} solution")
            record = {
                "action": "dissolve",
                "vessel": target_vessel,
                "solute": dissolve["solute"],
                "amount": dissolve["amount"],
                "unit": dissolve["unit"],
                "solvent": dissolve["solvent"],
                "volume": dissolve["volume"],
                "volume_units": dissolve["volume_units"],
                "ops": [
                    {"op": "add_solute", "vessel": target_vessel, "reagent": dissolve["solute"], "amount": dissolve["amount"], "unit": dissolve["unit"]},
                    *([
                        {"op": "add_solvent", "vessel": target_vessel, "reagent": comp["name"], "volume": comp["volume"], "volume_units": comp["volume_units"]}
                        for comp in (dissolve.get("solvents") or [{"name": dissolve["solvent"], "volume": dissolve["volume"], "volume_units": dissolve["volume_units"]}])
                    ]),
                    {"op": "stir", "vessel": target_vessel, "rpm": DEFAULTS["stir_rpm"], "minutes": 2}
                ],
                "raw": step,
                "reagents": [dissolve["solute"]] + [comp["name"] for comp in (dissolve.get("solvents") or [{"name": dissolve["solvent"]}])]
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Isolate/filter
        isolate = detect_filter_isolate(step)
        if isolate:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "isolate",
                "vessel": target_vessel,
                "ops": [{"op": "filter", "vessel": target_vessel}, {"op": "collect", "vessel": target_vessel}],
                "raw": step,
                "reagents": []
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Solution preparation
        prep = detect_solution_prep(step)
        if prep:
            vol_ml = prep["volume"] * (0.001 if prep["volume_units"].lower() in ("µl","ul") else (1.0 if prep["volume_units"].lower()=="ml" else 1000.0))
            explicit = prep.get("hardware_hint")
            label = explicit if explicit else "Beaker"
            vid = vessels.ensure_glassware(label, prefer_capacity_ml=vol_ml, explicit_hardware_hint=explicit)
            vessels.map_contents(vid, f"{prep['solvent']} {prep['concentration']} {prep['concentration_units']} solution of {prep['solute']}")
            hw_id = vessels.vessel_hardware(vid)
            record = {
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
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Additions (with optional temp/rate/time)
        solv = detect_add_solvent(step)
        if solv:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "add_solvent",
                "vessel": target_vessel,
                "solvent": solv["solvent"],
                "volume": solv["volume"],
                "volume_units": solv["volume_units"],
                "reagents": [solv["solvent"]],
                "ops": [{"op": "add_solvent", "vessel": target_vessel, "solvent": solv["solvent"],
                        "volume": solv["volume"], "volume_units": solv["volume_units"]}],
                "raw": step
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        add = detect_add(step)
        if add:
            src_key = re.sub(r"^\bthe\b\s+","",add["source_name"], flags=re.I).strip()
            dst_key = re.sub(r"^\bthe\b\s+","",add["target_name"], flags=re.I).strip()
            src_vid = vessels.ensure_glassware(src_key)
            dst_vid = vessels.ensure_glassware(dst_key)
            record = {
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
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Stirring
        st = detect_stir(step)
        if st:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action":"stir","vessel":target_vessel,"reagents":[],
                "minutes":st["minutes"], "temperature_C":st["temperature_C"], "rpm":st["rpm"],
                "ops":ops_for_stir(target_vessel, st["minutes"], st["rpm"], st["temperature_C"]), "raw": step
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Heating
        ht = detect_heat(step)
        if ht:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            temp = ht[0]["temperature_C"]; minutes = ht[1]["minutes"]
            record = {
                "action":"heat_hold","vessel":target_vessel,"reagents":[],
                "minutes": minutes, "temperature_C": temp,
                "ops": ops_for_heat(target_vessel, temp, minutes), "raw": step
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Cooling
        cl = detect_cool(step)
        if cl:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action":"cool_to","vessel":target_vessel,"reagents":[],
                "temperature_C": cl["temperature_C"],
                "ops": [{"op":"set_hotplate_temperature","hotplate_id":DEVICE_IDS["hotplate_id"],"temperature_C":cl["temperature_C"]}],
                "raw": step
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Sonication
        so = detect_sonicate(step)
        if so:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action":"sonicate","vessel":target_vessel,"reagents":[],
                "minutes": so["minutes"],
                "ops": [{"op":"sonicate","sonicator_id":DEVICE_IDS["sonicator_id"],"minutes":so["minutes"]}],
                "raw": step
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Filtration / washing / drying
        filt = detect_filter(step)
        if filt:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {"action":"postprocess","vessel":target_vessel,"reagents":[],"ops":ops_for_postproc(target_vessel,filt), "raw": step}
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        wd = detect_wash_dry(step)
        if wd:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {"action":"postprocess","vessel":target_vessel,"reagents":[],"ops":ops_for_postproc(target_vessel,wd), "raw": step}
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Resuspend
        res = detect_resuspend(step)
        if res:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "resuspend",
                "vessel": target_vessel,
                "ops": [{"op": "resuspend", "tube": f"{target_vessel}_tube"}],
                "raw": step,
                "reagents": []
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Collect
        col = detect_collect(step)
        if col:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "collect",
                "vessel": target_vessel,
                "ops": [{"op": "collect", "tube": f"{target_vessel}_tube"}],
                "raw": step,
                "reagents": []
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Discard
        dis = detect_discard(step)
        if dis:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "discard",
                "vessel": target_vessel,
                "ops": [{"op": "discard_supernatant", "tube": f"{target_vessel}_tube"}],
                "raw": step,
                "reagents": []
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Transfer
        tra = detect_transfer(step)
        if tra:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "transfer",
                "vessel": target_vessel,
                "ops": [{"op": "transfer", "to": tra["target"], "tube": f"{target_vessel}_tube"}],
                "raw": step,
                "reagents": []
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Fallback generic process node
        target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
        substeps = re.split(r"\band\b|;|\.", step)
        for sub in substeps:
            sub = sub.strip()
            if not sub: continue
            # Try all detectors again for each substep
            weigh = detect_weigh(sub)
            if weigh:
                record = {
                    "action": "weigh",
                    "vessel": target_vessel,
                    "reagent": weigh["reagent"],
                    "amount": weigh["amount"],
                    "unit": weigh["unit"],
                    "ops": [{"op": "weigh", "reagent": weigh["reagent"], "amount": weigh["amount"], "unit": weigh["unit"]}],
                    "raw": sub,
                    "reagents": [weigh["reagent"]]
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            transfer_exp = detect_transfer_explicit(sub)
            if transfer_exp:
                record = {
                    "action": "transfer",
                    "vessel": target_vessel,
                    "ops": [{"op": "transfer", "to": transfer_exp["target"], "tube": f"{target_vessel}_tube"}],
                    "raw": sub,
                    "reagents": []
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            dissolve = detect_dissolve(sub)
            if dissolve:
                record = {
                    "action": "dissolve",
                    "vessel": target_vessel,
                    "solute": dissolve["solute"],
                    "amount": dissolve["amount"],
                    "unit": dissolve["unit"],
                    "solvent": dissolve["solvent"],
                    "volume": dissolve["volume"],
                    "volume_units": dissolve["volume_units"],
                    "ops": [
                    {"op": "add_solute", "vessel": target_vessel, "reagent": dissolve["solute"], "amount": dissolve["amount"], "unit": dissolve["unit"]},
                    *([
                        {"op": "add_solvent", "vessel": target_vessel, "reagent": comp["name"], "volume": comp["volume"], "volume_units": comp["volume_units"]}
                        for comp in (dissolve.get("solvents") or [{"name": dissolve["solvent"], "volume": dissolve["volume"], "volume_units": dissolve["volume_units"]}])
                    ]),
                    {"op": "stir", "vessel": target_vessel, "rpm": DEFAULTS["stir_rpm"], "minutes": 2}
                ],
                    "raw": sub,
                    "reagents": [dissolve["solute"]] + [comp["name"] for comp in (dissolve.get("solvents") or [{"name": dissolve["solvent"]}])]
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            isolate = detect_filter_isolate(sub)
            if isolate:
                record = {
                    "action": "isolate",
                    "vessel": target_vessel,
                    "ops": [{"op": "filter", "vessel": target_vessel}, {"op": "collect", "vessel": target_vessel}],
                    "raw": sub,
                    "reagents": []
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            # ...existing fallback detectors (resuspend, collect, discard, transfer)...
            res = detect_resuspend(sub)
            if res:
                record = {
                    "action": "resuspend",
                    "vessel": target_vessel,
                    "ops": [{"op": "resuspend", "tube": f"{target_vessel}_tube"}],
                    "raw": sub,
                    "reagents": []
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            col = detect_collect(sub)
            if col:
                record = {
                    "action": "collect",
                    "vessel": target_vessel,
                    "ops": [{"op": "collect", "tube": f"{target_vessel}_tube"}],
                    "raw": sub,
                    "reagents": []
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            dis = detect_discard(sub)
            if dis:
                record = {
                    "action": "discard",
                    "vessel": target_vessel,
                    "ops": [{"op": "discard_supernatant", "tube": f"{target_vessel}_tube"}],
                    "raw": sub,
                    "reagents": []
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            tra = detect_transfer(sub)
            if tra:
                record = {
                    "action": "transfer",
                    "vessel": target_vessel,
                    "ops": [{"op": "transfer", "to": tra["target"], "tube": f"{target_vessel}_tube"}],
                    "raw": sub,
                    "reagents": []
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            # If still nothing, add as process
            record = {"action": "process", "vessel": target_vessel, "reagents": [], "ops": [], "raw": sub}
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)


    # Build micro-ops per step and a flattened micro plan
    for rec in records:
        rec["micro_ops"] = expand_ops_to_micro(rec.get("ops", []), vessels, hardware)

    micro_plan = []
    for i, rec in enumerate(records, 1):
        # --- 1) Expand any material transfer into primitive robot actions ---
        # Any 'add_*' step is already represented as an explicit 'transfer' op in rec['ops'].
        # We convert that to: pick_up (source) → pour (source→target) → place (source down)
        for op in rec.get("ops", []):
            if op.get("op") == "transfer":
                src_id = op.get("from")
                dst_id = op.get("to")
                if src_id and dst_id:
                    src_label = _label_for_vessel(src_id, vessels, hardware)
                    dst_label = _label_for_vessel(dst_id, vessels, hardware)
                    ctx = rec.get("vessel") or rec.get("target_vessel") or rec.get("source_vessel")

                    micro_plan.append({
                        "verb": "pick_up",
                        "object": src_label,
                        "step_index": i,
                        "context_vessel": ctx,
                    })
                    pour_item = {
                        "verb": "pour",
                        "from": src_label,
                        "to": dst_label,
                        "step_index": i,
                        "context_vessel": ctx,
                    }
                    rate = op.get("rate")
                    if rate:  # include if present
                        pour_item["rate"] = rate
                    micro_plan.append(pour_item)
                    micro_plan.append({
                        "verb": "place",
                        "object": src_label,
                        "to": "bench",
                        "step_index": i,
                        "context_vessel": ctx,
                    })

        # --- 2) Copy through any authored micro_ops EXCEPT 'note add_*' markers ---
        for micro in rec.get("micro_ops", []):
            # Drop the diagnostic "note/op=add_*" entries so nothing shows up as a 'set'
            if (
                micro.get("device") == "note"
                and micro.get("param") == "op"
                and str(micro.get("value", "")).startswith("add_")
            ):
                continue
            item = dict(micro)
            item["step_index"] = i
            item["context_vessel"] = rec.get("vessel") or rec.get("target_vessel") or rec.get("source_vessel")
            micro_plan.append(item)

    return {
        "hardware": hardware,
        "vessel_registry": vessels.as_dict(),
        "vessel_contents": vessels.contents_dict(),
        "devices": DEVICE_IDS,
        "micro_plan": micro_plan,
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
        if ":" not in line:
            raise ValueError(f"line {lineno}: missing ':' separator")
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
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
        print(f"Wrote {args.out}")