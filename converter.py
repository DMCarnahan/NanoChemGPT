from __future__ import annotations

import copy
import json
import pathlib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


DEFAULTS: Dict[str, Any] = {
    "stir_rpm": 700,
    "centrifuge_rpm": 4000,
    "centrifuge_minutes": 10,
    "transfer_rate_slow": "slow",
    "room_temp_C": 25.0,
}

DEVICE_IDS: Dict[str, str] = {
    "stir_plate_id": "SP1",
    "hotplate_id": "HP1",
    "centrifuge_id": "CF1",
    "oven_id": "OV1",
    "vacuum_pump_id": "VP1",
    "sonicator_id": "US1",
    "ph_meter_id": "PH1",
    "autotitrator_id": "AT1",
}

INLINE_TAG_RX = re.compile(r"\s*\[(?:CTX|DB|PARSED|GEN|\d+)\]\s*", re.I)
FENCE_START_RX = re.compile(r"^\s*```")
NON_PROC_HEAD_RX = re.compile(
    r"^\s*#{1,6}\s*(references?|sources?|bibliography|rationale|reasoning|notes|discussion|supplementary|appendix|acknowledge?ments?)\b",
    re.I,
)

_UNIT_CANON = {
    "l": "L",
    "L": "L",
    "ml": "mL",
    "mL": "mL",
    "ul": "µL",
    "uL": "µL",
    "µl": "µL",
    "µL": "µL",
    "g": "g",
    "mg": "mg",
    "µg": "µg",
    "ug": "µg",
    "kg": "kg",
    "mol": "mol",
    "mmol": "mmol",
    "µmol": "µmol",
    "umol": "µmol",
    "M": "M",
    "m": "M",
    "mM": "mM",
    "mm": "mM",
    "µM": "µM",
    "uM": "µM",
    "wt%": "wt%",
    "vol%": "vol%",
}

_AMOUNT_UNIT = r"(?:~?\d+(?:[.\u2013\u2014-]\d+)?\s*(?:µ?u?L|mL|ml|L|l|mg|g|kg|µg|ug|mol|mmol|µmol|umol)\b)"
SPLIT_BOUNDARY_RX = re.compile(
    rf"\s*(?:,|\+|\band\b|\balong with\b|\btogether with\b)\s*(?=(?:{_AMOUNT_UNIT}))",
    re.I,
)
_AMOUNT_RE = r"(?P<approx>[~≈])?\s*(?P<val>\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?)\s*(?P<unit>µ?u?L|mL|ml|L|l|mg|g|kg|µg|ug|mol|mmol|µmol|umol)\b"
_PAREN_AMOUNT_RX = re.compile(r"\(\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>µ?u?L|mL|ml|L|l|mg|g|kg|µg|ug|mol|mmol|µmol|umol)\s*\)")
_CONC_RX = re.compile(
    r"(?P<approx>[~≈])?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|µM|uM)\s+(?P<name>[^(),;]+(?:\s+[^(),;]+)*)(?:\s*\([^)]*\))?(?:\s+in\s+(?P<solvent>[^(),;]+))?\s*(?:\bsolution\b)?",
    re.I,
)
_LEAD_AMT_RX = re.compile(rf"{_AMOUNT_RE}" + r"\s*(?:of\s+)?(?P<name>.+?)\s*(?P<paren>\([^)]*\))?\s*$", re.I)
_APPROX_WORD_RX = re.compile(r"\b(about|approximately|approx\.)\b", re.I)

DEFAULT_HARDWARE: List[Dict[str, Any]] = [
    {"id": "H1", "name": "Magnetic stirrer", "type": "hardware", "capacity": None},
    {"id": "H2", "name": "Centrifuge", "type": "hardware", "capacity": None},
    {"id": "H3", "name": "Centrifuge tubes", "type": "hardware", "capacity": None},
    {"id": "H4", "name": "Beaker", "type": "beaker", "capacity": None},
    {"id": "H5", "name": "Pipettes", "type": "hardware", "capacity": None},
]


def _clean_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return s.replace("° ", "°").replace("–", "-").replace("—", "-")


def strip_tags(s: str) -> str:
    s = _clean_unicode(s)
    s = re.sub(r"`{3,}.*$", "", s)
    s = INLINE_TAG_RX.sub(" ", s)
    s = re.sub(r"</?[^>]+>", "", s)
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _canon_unit(u: str) -> str:
    return _UNIT_CANON.get((u or "").strip(), (u or "").strip())


def _to_float_range(s: str) -> tuple[Optional[float], Optional[float]]:
    s = (s or "").replace("–", "-")
    if "-" in s and not s.startswith("-"):
        a, b = s.split("-", 1)
        try:
            return float(a), float(b)
        except Exception:
            pass
    try:
        return float(s), None
    except Exception:
        return None, None


def split_reagent_phrases(text: str) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    return [p.strip().strip(",") for p in re.split(SPLIT_BOUNDARY_RX, s) if p.strip().strip(",")]


def parse_reagent_phrase_to_struct(s: str) -> Dict[str, Any]:
    original = s
    s = strip_tags((s or "").strip())
    approx = bool(_APPROX_WORD_RX.search(s))
    is_solution = bool(re.search(r"\bsolution\b", s, re.I))
    display_name = original

    m = _CONC_RX.search(s)
    if m:
        return {
            "name": m.group("name").strip(),
            "amount": None,
            "amount_unit": None,
            "amount_range": None,
            "alt_amount": None,
            "alt_unit": None,
            "concentration": float(m.group("val")),
            "conc_unit": _canon_unit(m.group("unit")),
            "solvent": (m.group("solvent") or "").strip() or None,
            "approx": approx or bool(m.group("approx")),
            "original": original,
            "is_solution": True,
            "display_name": display_name,
        }

    m = _LEAD_AMT_RX.match(s)
    if m:
        low, high = _to_float_range(m.group("val"))
        amount_range = [low, high] if high is not None else None
        alt_amount = None
        alt_unit = None
        paren = m.group("paren") or ""
        pm = _PAREN_AMOUNT_RX.search(paren)
        if pm:
            alt_amount = float(pm.group("val"))
            alt_unit = _canon_unit(pm.group("unit"))
        return {
            "name": (m.group("name") or "").strip().strip(",;"),
            "amount": low,
            "amount_unit": _canon_unit(m.group("unit")),
            "amount_range": amount_range,
            "alt_amount": alt_amount,
            "alt_unit": alt_unit,
            "concentration": None,
            "conc_unit": None,
            "solvent": None,
            "approx": approx or bool(m.group("approx")),
            "original": original,
            "is_solution": is_solution,
            "display_name": display_name,
        }

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
        "original": original,
        "is_solution": is_solution,
        "display_name": display_name,
    }


def _normalize_reagents_inplace(record: Dict[str, Any]) -> None:
    reag = record.get("reagents", [])
    flat: List[str] = []
    for item in (reag if isinstance(reag, list) else [reag]):
        if isinstance(item, str):
            flat.extend(split_reagent_phrases(item))
        elif item:
            flat.append(item)
    record["reagents"] = flat

    solute = record.get("solute", "")
    if isinstance(solute, str) and solute.strip():
        record["solutes"] = split_reagent_phrases(solute)


def _add_structured_reagents_inplace(record: Dict[str, Any]) -> None:
    reag = record.get("reagents", []) or []
    if not isinstance(reag, list):
        reag = [reag]
    record["reagents_structured"] = [
        parse_reagent_phrase_to_struct(x) for x in reag if isinstance(x, str) and x.strip()
    ]

    solutes = record.get("solutes", []) or []
    if isinstance(solutes, list) and solutes:
        record["solutes_structured"] = [
            parse_reagent_phrase_to_struct(x) for x in solutes if isinstance(x, str) and x.strip()
        ]


def find_temp_c(text: str) -> Optional[float]:
    s = _clean_unicode(text)
    if re.search(r"\breflux\b", s, re.I):
        return 100.0
    if re.search(r"\bboil(?:ing)?\b", s, re.I):
        return 100.0
    if re.search(r"\bice\s*bath\b", s, re.I):
        return 0.0
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CFK])\b", s, re.I)
    if not m:
        if re.search(r"\b(rt|room\s*temp(?:erature)?)\b", s, re.I):
            return DEFAULTS["room_temp_C"]
        return None
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "C":
        return val
    if unit == "F":
        return (val - 32.0) * 5.0 / 9.0
    if unit == "K":
        return val - 273.15
    return None


def find_minutes(text: str) -> Optional[float]:
    s = _clean_unicode(text)
    found = False
    minutes = 0.0
    if re.search(r"\bover\s*night\b", s, re.I):
        return 12 * 60.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:second|sec|s)\b", s, re.I):
        minutes += float(m.group(1)) / 60.0
        found = True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:hour|hr|hrs|h)\b", s, re.I):
        minutes += float(m.group(1)) * 60.0
        found = True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", s, re.I):
        minutes += float(m.group(1))
        found = True
    return minutes if found else None


def parse_hardware(markdown_text: str) -> List[Dict[str, Any]]:
    lines = markdown_text.splitlines()
    items: List[Dict[str, Any]] = []
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
                m = re.match(r"(Beakers?|Flasks?)\s*\((.+?)\)", entry, re.I)
                if m:
                    base = "beaker" if "beaker" in m.group(1).lower() else "flask"
                    parts = re.split(r"\s*(?:and|,)\s*", m.group(2))
                    for part in parts:
                        items.append(
                            {
                                "name": f"{m.group(1).split()[0].title()} {part.strip()}",
                                "type": base,
                                "capacity": part.strip(),
                            }
                        )
                else:
                    capm = re.search(r"(\d+)\s*(µ?u?L|mL|L)\b", entry, re.I)
                    cap = capm.group(0) if capm else None
                    typ = "beaker" if "beaker" in entry.lower() else ("flask" if "flask" in entry.lower() else "hardware")
                    nm = entry if typ == "hardware" else (f"{typ.title()} {cap}" if cap else typ.title())
                    items.append({"name": nm, "type": typ, "capacity": cap})
    if not items:
        return copy.deepcopy(DEFAULT_HARDWARE)
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        row = dict(item)
        row["id"] = f"H{i}"
        out.append(row)
    return out


def _capacity_to_ml(cap: Optional[str]) -> Optional[float]:
    if not cap:
        return None
    m = re.match(r"(\d+(?:\.\d+)?)\s*(µ?u?L|mL|L)\b", cap, re.I)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("µl", "ul"):
        return val / 1000.0
    if unit == "ml":
        return val
    return val * 1000.0


class VesselRegistry:
    def __init__(self, hardware: List[Dict[str, Any]]) -> None:
        self.hardware = hardware
        self._counter = 0
        self.primary_vessel: Optional[str] = None
        self._vid_to_label: Dict[str, str] = {}
        self._label_to_vid: Dict[str, str] = {}
        self._vid_to_hid: Dict[str, str] = {}
        self.contents_summary: Dict[str, str] = {}
        self.contents_detailed: Dict[str, Any] = {}

    def _new_vid(self) -> str:
        self._counter += 1
        return f"V{self._counter}"

    def _pick_glass_for_volume(self, vol_ml: Optional[float]) -> Optional[Dict[str, Any]]:
        choices = [h for h in self.hardware if h.get("type") in {"beaker", "flask"}]
        if not choices:
            return None
        if vol_ml is None:
            return sorted(choices, key=lambda h: (_capacity_to_ml(h.get("capacity")) or 1e9))[0]
        target = vol_ml * 1.5
        candidates = [(h, _capacity_to_ml(h.get("capacity")) or 1e12) for h in choices]
        viable = [pair for pair in candidates if pair[1] >= target]
        if viable:
            return sorted(viable, key=lambda x: x[1])[0][0]
        return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]

    def ensure_glassware(
        self,
        label: str,
        *,
        prefer_capacity_ml: Optional[float] = None,
        explicit_hardware_hint: Optional[str] = None,
    ) -> str:
        key = (label or "Beaker").lower().strip()
        if key in self._label_to_vid:
            return self._label_to_vid[key]
        vid = self._new_vid()
        hw_id = None
        if explicit_hardware_hint:
            for h in self.hardware:
                if explicit_hardware_hint.lower() in h["name"].lower():
                    hw_id = h["id"]
                    break
        if hw_id is None:
            chosen = self._pick_glass_for_volume(prefer_capacity_ml)
            if chosen:
                hw_id = chosen["id"]
        self._vid_to_label[vid] = label or "Beaker"
        self._label_to_vid[key] = vid
        if hw_id:
            self._vid_to_hid[vid] = hw_id
        if self.primary_vessel is None:
            self.primary_vessel = vid
        return vid

    def vessel_hardware(self, vid: str) -> Optional[str]:
        return self._vid_to_hid.get(vid)

    def as_dict(self) -> Dict[str, str]:
        return dict(self._vid_to_label)


def extract_steps(markdown_text: str) -> List[str]:
    lines = markdown_text.splitlines()
    in_proc = False
    steps: List[str] = []
    buf: List[str] = []

    for line in lines:
        if re.search(r"(?:\*\*)?Procedure:?(?:\*\*)?", line, re.I):
            in_proc = True
            continue

        if in_proc:
            if FENCE_START_RX.match(line) or NON_PROC_HEAD_RX.match(line):
                break
            if re.match(r"\s*\d+\.\s", line):
                if buf:
                    steps.append(" ".join(buf).strip())
                    buf = []
                buf.append(re.sub(r"^\s*\d+\.\s*", "", line).strip())
            else:
                if line.strip() and not line.strip().startswith("```"):
                    buf.append(line.strip())

    if buf:
        steps.append(" ".join(buf).strip())

    if not steps and markdown_text.strip():
        cleaned = strip_tags(markdown_text.strip())
        if cleaned:
            return [cleaned]

    return [strip_tags(s) for s in steps if s.strip()]


_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def _parse_repeat_count(text: str, default: int = 1) -> int:
    s = (text or "").lower()
    m = re.search(r"(\d+)\s*(?:x|×|times?|washes?)", s)
    if m:
        return int(m.group(1))
    for word, value in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b\s+times?", s) or re.search(rf"\b{word}\b\s+washes?\b", s):
            return value
    return default


def _split_substeps(step: str) -> List[str]:
    return [x.strip() for x in re.split(r"\band\b|;|(?<!\d)\.(?!\d)", step) if x.strip()]


def _clean_solvent_tail(solvent: str) -> str:
    solvent = strip_tags((solvent or "").strip().rstrip(",."))
    solvent = solvent.split(" in ")[0].strip()
    return solvent


def _slug_token(s: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def _extract_concentration_components(text: str) -> List[Dict[str, Any]]:
    comps: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|µM|uM)\s+of\s+(?P<name>[^,.;]+?)(?=(?:\s+and\s+\d+(?:\.\d+)?\s*(?:M|mM|µM|uM)\s+of\s+)|(?:\s+in\s+)|$)",
        re.I,
    )
    for m in pattern.finditer(text):
        comps.append(
            {
                "name": m.group("name").strip(),
                "concentration": float(m.group("val")),
                "conc_unit": _canon_unit(m.group("unit")),
            }
        )
    return comps


def detect_prepare_solution_components(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\b(dissolv\w*|prepare)\b", s, re.I):
        return None
    if not re.search(r"\bto\s+prepare\b|\bprecursor solution\b|\bmetal precursor solution\b", s, re.I):
        return None

    solvent_match = re.search(r"\bin\s+([A-Za-z][^.;]+?)(?:\s+to\s+prepare|$)", s, re.I)
    components = _extract_concentration_components(s)
    if not components or not solvent_match:
        return None

    solvent = _clean_solvent_tail(solvent_match.group(1))
    return {
        "action": "prepare_solution",
        "components": components,
        "solvent": solvent,
        "description": f"metal precursor solution in {solvent}",
    }


def detect_prepare_and_add_solution(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    m = re.search(
        r"\bprepare\s+a\s+(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|µM|uM)\s+solution\s+of\s+(?P<reagent>[^,.;]+?)\s+in\s+(?P<solvent>[^,.;]+?)\s+and\s+(?:combine|add|mix)\s+(?:it\s+)?with\b",
        s,
        re.I,
    )
    if not m:
        return None
    return {
        "action": "add_prepared_solution",
        "reagent": m.group("reagent").strip(),
        "solvent": _clean_solvent_tail(m.group("solvent")),
        "concentration": float(m.group("val")),
        "conc_unit": _canon_unit(m.group("unit")),
        "with_stirring": bool(re.search(r"\bstirr", s, re.I)),
    }


def detect_add_solvent(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    m = re.search(
        r"\badd\s+(?P<vol>\d+(?:\.\d+)?)\s*(?P<vunit>µ?u?L|mL|ml|L|l)\s+of\s+(?P<solvent>.+?)\s+to\s+(?:the\s+)?(?:solution|mixture|suspension|dispersion)\b",
        s,
        re.I,
    )
    if not m:
        return None
    return {
        "action": "add_solvent",
        "solvent": m.group("solvent").strip(),
        "volume": float(m.group("vol")),
        "volume_units": m.group("vunit"),
        "minutes": find_minutes(s),
        "with_stirring": bool(re.search(r"\bstirr", s, re.I)),
    }


def detect_prepare_and_add_reagent_solution(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    m = re.search(
        r"\bprepare\s+an?\s+(?P<solvent>[^\s]+)\s+solution\s+of\s+(?P<reagent>[^,.;]+?)\s+and\s+add\s+it\s+to\b",
        s,
        re.I,
    )
    if not m:
        return None
    return {
        "action": "add_reagent_solution",
        "reagent": m.group("reagent").strip(),
        "solvent": m.group("solvent").strip(),
        "with_stirring": bool(re.search(r"\bstirr", s, re.I)),
    }


def detect_stir(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\bstir", s, re.I):
        return None
    m_rpm = re.search(r"(\d{2,5})\s*rpm\b", s, re.I)
    rpm = int(m_rpm.group(1)) if m_rpm else DEFAULTS["stir_rpm"]
    minutes = find_minutes(s)
    temp = find_temp_c(s)
    return {
        "action": "stir",
        "rpm": rpm,
        "minutes": minutes if minutes is not None else 60.0,
        "temperature_C": temp if temp is not None else DEFAULTS["room_temp_C"],
    }


def detect_explicit_postprocess(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\bcentrifug", s, re.I) and not re.search(r"\bwash\b", s, re.I):
        return None

    solvent_match = re.search(
        r"with\s+(?P<vol>\d+(?:\.\d+)?)\s*(?P<vunit>µ?u?L|mL|ml|L|l)\s+of\s+(?P<solvent>[^,.;]+)",
        s,
        re.I,
    )
    centrifuge_match = re.search(
        r"centrifug\w*.*?(?P<mins>\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\s+at\s+(?P<rpm>\d+)\s*rpm",
        s,
        re.I,
    )
    if not solvent_match and not centrifuge_match:
        return None

    return {
        "action": "postprocess",
        "wash_count": _parse_repeat_count(s, default=1),
        "wash_solvent": _clean_solvent_tail(solvent_match.group("solvent")) if solvent_match else "wash solvent",
        "wash_volume": float(solvent_match.group("vol")) if solvent_match else None,
        "wash_volume_units": solvent_match.group("vunit") if solvent_match else None,
        "centrifuge_minutes": float(centrifuge_match.group("mins")) if centrifuge_match else DEFAULTS["centrifuge_minutes"],
        "centrifuge_rpm": int(centrifuge_match.group("rpm")) if centrifuge_match else DEFAULTS["centrifuge_rpm"],
    }


def detect_redisperse(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\b(disperse|redisperse)\b", s, re.I):
        return None
    m = re.search(r"\bin\s+(?P<solvent>[^,.;]+)", s, re.I)
    solvent = _clean_solvent_tail(m.group("solvent")) if m else None
    return {"action": "redisperse", "solvent": solvent}


def detect_solution_prep(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    patterns = [
        re.compile(
            r"prepare\s+a\s+([\d\.]+)\s*(M|mM|µM|uM)\s+.+?\s+solution\s+by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|L|l)\s+of\s+(?P<solvent>.+?)\s*(?:in\b|$)",
            re.I,
        ),
        re.compile(
            r"dissolv\w*\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|L|l)\s+of\s+(?P<solvent>.+?)\s+to\s+(?:make|form|yield|obtain)\s+a\s+([\d\.]+)\s*(M|mM|µM|uM|%)\s+.+?\s+solution",
            re.I,
        ),
    ]
    for rx in patterns:
        m = rx.search(s)
        if m:
            conc_match = re.search(r"(\d+(?:\.\d+)?)\s*(M|mM|µM|uM|%)\s+.+?\s+solution", s, re.I)
            return {
                "action": "prepare_solution",
                "solute": m.group("solute").strip(),
                "solvent": _clean_solvent_tail(m.group("solvent")),
                "concentration": float(conc_match.group(1)) if conc_match else None,
                "concentration_units": _canon_unit(conc_match.group(2)) if conc_match else None,
                "volume": float(m.group("vol")),
                "volume_units": m.group("vunit"),
                "hardware_hint": None,
            }
    return None


def detect_dissolve(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    m = re.search(
        r"\bdissolv\w*\s+(?P<amount>[\d\.]+)\s*(?P<unit>mg|g|kg|µg|ug)\s+(?:of\s+)?(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|L|l)\s+(?:of\s+)?(?P<solvent>[^.;,]+)",
        s,
        re.I,
    )
    if not m:
        return None

    solvent_captured = _clean_solvent_tail(m.group("solvent").strip())
    vol1 = float(m.group("vol"))
    vunit1 = m.group("vunit")
    extras: List[Dict[str, Any]] = []
    inline = solvent_captured
    while True:
        exm = re.search(
            r"(.*?)(?:,\s*)?(?:and\s+)([\d\.]+)\s*(µ?u?L|mL|ml|L|l)\s+of\s+([^,;]+)$",
            inline,
            re.I,
        )
        if not exm:
            break
        base = exm.group(1).strip()
        extras.insert(
            0,
            {
                "name": _clean_solvent_tail(exm.group(4).strip()),
                "volume": float(exm.group(2)),
                "volume_units": exm.group(3),
            },
        )
        inline = base
    solvent1 = inline

    result: Dict[str, Any] = {
        "action": "dissolve",
        "solute": m.group("solute").strip(),
        "amount": float(m.group("amount")),
        "unit": m.group("unit"),
        "solvent": solvent1 if not extras else solvent1 + " + " + " + ".join(e["name"] for e in extras),
        "volume": vol1,
        "volume_units": vunit1,
    }
    if extras:
        result["solvents"] = [{"name": solvent1, "volume": vol1, "volume_units": vunit1}] + extras
    return result


def detect_transfer(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    m = re.search(r"\btransfer\b.*\b(?:into|to)\b\s+(?P<target>.+)$", s, re.I)
    if not m:
        return None
    return {"action": "transfer", "target": m.group("target").strip()}


def detect_weigh(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    m = re.search(r"\bweigh\s+(?P<amount>[\d\.]+)\s*(?P<unit>mg|g|kg|µg|ug)\s+of\s+(?P<reagent>.+)$", s, re.I)
    if not m:
        return None
    return {
        "action": "weigh",
        "reagent": m.group("reagent").strip(),
        "amount": float(m.group("amount")),
        "unit": m.group("unit"),
    }


def detect_generic_add(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    m = re.search(r"\b(add|charge)\s+(?:the\s+)?(?P<src>.+?)\s+to\s+(?:the\s+)?(?P<dst>.+?)\b", s, re.I)
    if not m:
        return None
    src = m.group("src").strip()
    if src.lower() in {"it", "this", "that"}:
        return None
    return {
        "action": "add",
        "source_name": src,
        "target_name": m.group("dst").strip(),
        "rate": "slow" if re.search(r"\b(dropwise|slow)\b", s, re.I) else "normal",
        "temperature_C": find_temp_c(s),
        "minutes": find_minutes(s),
    }


def semantic_parse_step(step: str) -> Dict[str, Any]:
    detectors = [
        detect_prepare_solution_components,
        detect_prepare_and_add_solution,
        detect_add_solvent,
        detect_prepare_and_add_reagent_solution,
        detect_explicit_postprocess,
        detect_redisperse,
        detect_stir,
        detect_solution_prep,
        detect_dissolve,
        detect_weigh,
        detect_transfer,
        detect_generic_add,
    ]
    for detector in detectors:
        parsed = detector(step)
        if parsed:
            parsed["raw"] = step
            return parsed

    target_vessel = "V1"
    record: Dict[str, Any] = {"action": "process", "vessel": target_vessel, "reagents": [], "ops": [], "raw": step}
    substeps = _split_substeps(step)
    if len(substeps) > 1:
        fragments: List[Dict[str, Any]] = []
        for sub in substeps:
            parsed = semantic_parse_step(sub)
            if parsed.get("action") != "process":
                fragments.append(parsed)
        if fragments:
            return {"action": "composite", "parts": fragments, "raw": step}
    return record


def semantic_parse(text: str, vessels: VesselRegistry) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    steps = extract_steps(text)
    records: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {"prepared_tokens": {}}

    for step in steps:
        parsed = semantic_parse_step(step)
        if parsed.get("action") == "composite":
            for part in parsed["parts"]:
                records.append(part)
            continue
        records.append(parsed)

    return records, context


def _ensure_primary_vessel(vessels: VesselRegistry) -> str:
    return vessels.primary_vessel or vessels.ensure_glassware("Beaker")


def _emit_prepare_solution(step: Dict[str, Any], vessel: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    components = step["components"]
    solvent = step["solvent"]
    ops: List[Dict[str, Any]] = []
    reagents: List[str] = []
    detailed_components: List[Dict[str, Any]] = []

    for comp in components:
        ops.append({"op": "pour", "reagent": comp["name"], "vessel": vessel})
        reagents.append(comp["name"])
        detailed_components.append(
            {
                "name": comp["name"],
                "display_name": comp["name"],
                "concentration": comp["concentration"],
                "conc_unit": comp["conc_unit"],
                "solvent": solvent,
            }
        )
    ops.append({"op": "pour", "solvent": solvent, "vessel": vessel})
    reagents.append(solvent)
    detail = {
        "description": step.get("description", f"prepared solution in {solvent}"),
        "components": detailed_components + [{"name": solvent, "display_name": solvent, "role": "solvent"}],
    }
    record = {
        "action": "prepare_solution",
        "raw": step["raw"],
        "vessel": vessel,
        "ops": ops,
        "reagents": reagents,
    }
    return record, detail


def _emit_add_prepared_solution(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    reagent_token = f"{_slug_token(step['reagent'])}_solution_in_{_slug_token(step['solvent'])}"
    ops = [
        {"op": "move_to_stir_plate", "stir_plate_id": DEVICE_IDS["stir_plate_id"], "vessel": vessel},
        {"op": "set_stir_rate", "vessel": vessel, "rpm": DEFAULTS["stir_rpm"], "inferred": True},
        {"op": "pour", "reagent": reagent_token, "vessel": vessel},
    ]
    return {
        "action": "add",
        "raw": step["raw"],
        "vessel": vessel,
        "rpm": DEFAULTS["stir_rpm"],
        "ops": ops,
        "reagents": [step["reagent"], step["solvent"]],
    }


def _emit_add_solvent(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    ops: List[Dict[str, Any]] = [
        {
            "op": "pour",
            "reagent": step["solvent"],
            "vessel": vessel,
            "volume": step["volume"],
            "volume_units": step["volume_units"],
        }
    ]
    if step.get("minutes") is not None:
        ops.append({"op": "wait", "minutes": step["minutes"]})
    return {
        "action": "add_solvent",
        "raw": step["raw"],
        "vessel": vessel,
        "solvent": step["solvent"],
        "volume": step["volume"],
        "volume_units": step["volume_units"],
        "ops": ops,
        "reagents": [step["solvent"]],
    }


def _emit_add_reagent_solution(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    token = f"{_slug_token(step['reagent'])}_{_slug_token(step['solvent'])}_solution"
    ops = [{"op": "pour", "reagent": token, "vessel": vessel}]
    return {
        "action": "add",
        "raw": step["raw"],
        "vessel": vessel,
        "with_stirring": step.get("with_stirring", True),
        "ops": ops,
        "reagents": [f"{step['reagent']} aqueous solution" if step["solvent"].lower() == "aqueous" else f"{step['reagent']} {step['solvent']} solution"],
    }


def _emit_stir(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    return {
        "action": "stir",
        "raw": step["raw"],
        "vessel": vessel,
        "rpm": step["rpm"],
        "minutes": step["minutes"],
        "ops": [{"op": "wait", "minutes": step["minutes"]}],
        "reagents": [],
        "reagents_structured": [],
    }


def _emit_postprocess(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    ops: List[Dict[str, Any]] = [
        {"op": "transfer_to_centrifuge_tube", "from": vessel, "to": f"{vessel}_tube"},
        {
            "op": "centrifuge",
            "centrifuge_id": DEVICE_IDS["centrifuge_id"],
            "tube": f"{vessel}_tube",
            "minutes": step["centrifuge_minutes"],
            "rpm": step["centrifuge_rpm"],
            "inferred": True,
        },
        {
            "op": "decant_supernatant",
            "tube": f"{vessel}_tube",
            "executable": False,
            "review_required": True,
            "review_reason": "unsupported_phase_separation_primitive",
        },
    ]
    for wash_cycle in range(1, step["wash_count"] + 1):
        ops.append(
            {
                "op": "pour",
                "reagent": step["wash_solvent"],
                "vessel": f"{vessel}_tube",
                "volume": step["wash_volume"],
                "volume_units": step.get("wash_volume_units") or "mL",
                "wash_cycle": wash_cycle,
            }
        )
        ops.append(
            {
                "op": "resuspend",
                "tube": f"{vessel}_tube",
                "wash_cycle": wash_cycle,
                "executable": False,
                "review_required": True,
                "review_reason": "unsupported_dispersion_primitive",
            }
        )
        ops.append(
            {
                "op": "centrifuge",
                "centrifuge_id": DEVICE_IDS["centrifuge_id"],
                "tube": f"{vessel}_tube",
                "minutes": step["centrifuge_minutes"],
                "rpm": step["centrifuge_rpm"],
                "wash_cycle": wash_cycle,
            }
        )
        ops.append(
            {
                "op": "decant_supernatant",
                "tube": f"{vessel}_tube",
                "wash_cycle": wash_cycle,
                "executable": False,
                "review_required": True,
                "review_reason": "unsupported_phase_separation_primitive",
            }
        )

    return {
        "action": "postprocess",
        "raw": step["raw"],
        "vessel": vessel,
        "ops": ops,
        "reagents": [step["wash_solvent"]],
        "reagents_structured": [
            {
                "name": step["wash_solvent"],
                "display_name": step["wash_solvent"].title() if step["wash_solvent"].islower() else step["wash_solvent"],
                "original": f"{step['wash_volume']} {step.get('wash_volume_units') or 'mL'} of {step['wash_solvent']}",
                "amount": step["wash_volume"],
                "amount_unit": step.get("wash_volume_units") or "mL",
                "repetitions": step["wash_count"],
            }
        ],
    }


def _emit_redisperse(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    redisperse_vessel = f"{vessel}_tube"
    solvent = step.get("solvent") or "solvent"
    return {
        "action": "redisperse",
        "raw": step["raw"],
        "vessel": redisperse_vessel,
        "ops": [
            {"op": "pour", "reagent": solvent, "vessel": redisperse_vessel},
            {
                "op": "resuspend",
                "tube": redisperse_vessel,
                "executable": False,
                "review_required": True,
                "review_reason": "unsupported_dispersion_primitive",
            },
        ],
        "reagents": [solvent],
    }


def _emit_solution_prep(step: Dict[str, Any], vessels: VesselRegistry) -> Dict[str, Any]:
    vol_ml = step["volume"] * (
        0.001 if step["volume_units"].lower() in ("µl", "ul") else (1.0 if step["volume_units"].lower() == "ml" else 1000.0)
    )
    explicit = step.get("hardware_hint")
    label = explicit if explicit else "Beaker"
    vid = vessels.ensure_glassware(label, prefer_capacity_ml=vol_ml, explicit_hardware_hint=explicit)
    ops = [
        {"op": "pour", "reagent": step["solute"], "vessel": vid},
        {"op": "pour", "solvent": step["solvent"], "vessel": vid, "volume": step["volume"], "volume_units": step["volume_units"]},
    ]
    record = {
        "action": "prepare_solution",
        "raw": step["raw"],
        "vessel": vid,
        "solute": step["solute"],
        "solvent": step["solvent"],
        "concentration": step.get("concentration"),
        "concentration_units": step.get("concentration_units"),
        "volume": step["volume"],
        "volume_units": step["volume_units"],
        "reagents": [step["solute"], step["solvent"]],
        "ops": ops,
    }
    return record


def _emit_dissolve(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    solvents = step.get("solvents") or [{"name": step["solvent"], "volume": step["volume"], "volume_units": step["volume_units"]}]
    ops = [{"op": "add_solute", "vessel": vessel, "reagent": step["solute"], "amount": step["amount"], "unit": step["unit"]}]
    ops.extend(
        {"op": "add_solvent", "vessel": vessel, "reagent": s["name"], "volume": s["volume"], "volume_units": s["volume_units"]}
        for s in solvents
    )
    ops.append({"op": "stir", "vessel": vessel, "rpm": DEFAULTS["stir_rpm"], "minutes": 2})
    return {
        "action": "dissolve",
        "raw": step["raw"],
        "vessel": vessel,
        "solute": step["solute"],
        "amount": step["amount"],
        "unit": step["unit"],
        "solvent": step["solvent"],
        "volume": step["volume"],
        "volume_units": step["volume_units"],
        "reagents": [step["solute"]] + [s["name"] for s in solvents],
        "ops": ops,
    }


def _emit_weigh(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    return {
        "action": "weigh",
        "raw": step["raw"],
        "vessel": vessel,
        "reagent": step["reagent"],
        "amount": step["amount"],
        "unit": step["unit"],
        "reagents": [step["reagent"]],
        "ops": [{"op": "weigh", "reagent": step["reagent"], "amount": step["amount"], "unit": step["unit"]}],
    }


def _emit_transfer(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    return {
        "action": "transfer",
        "raw": step["raw"],
        "vessel": vessel,
        "reagents": [],
        "ops": [{"op": "transfer", "to": step["target"], "tube": f"{vessel}_tube"}],
    }


def _emit_generic_add(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    src = step["source_name"]
    dst = step["target_name"]
    src_vid = vessel if "mixture" in src.lower() or "solution" in src.lower() else vessel
    dst_vid = vessel if "mixture" in dst.lower() or "solution" in dst.lower() else vessel
    ops = [
        {"op": "move_to_stir_plate", "stir_plate_id": DEVICE_IDS["stir_plate_id"], "vessel": dst_vid},
        {"op": "set_stir_rate", "vessel": dst_vid, "rpm": DEFAULTS["stir_rpm"]},
        {"op": "transfer", "from": src_vid, "to": dst_vid, "rate": step["rate"]},
    ]
    if step.get("minutes") is not None:
        ops.append({"op": "wait", "minutes": step["minutes"]})
    return {
        "action": "add",
        "raw": step["raw"],
        "source_vessel": src_vid,
        "target_vessel": dst_vid,
        "reagents": [src],
        "with_stirring": True,
        "rate": step["rate"],
        "temperature_C": step.get("temperature_C"),
        "minutes": step.get("minutes"),
        "ops": ops,
    }


def _emit_fallback(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    return {"action": "process", "raw": step["raw"], "vessel": vessel, "reagents": [], "ops": []}


def emit_steps(semantic_steps: List[Dict[str, Any]], vessels: VesselRegistry) -> List[Dict[str, Any]]:
    emitted: List[Dict[str, Any]] = []
    primary = _ensure_primary_vessel(vessels)

    for idx, step in enumerate(semantic_steps, start=1):
        action = step["action"]
        detail_payload = None

        if action == "prepare_solution" and step.get("components"):
            record, detail_payload = _emit_prepare_solution(step, primary)
        elif action == "add_prepared_solution":
            record = _emit_add_prepared_solution(step, primary)
        elif action == "add_solvent":
            record = _emit_add_solvent(step, primary)
        elif action == "add_reagent_solution":
            record = _emit_add_reagent_solution(step, primary)
        elif action == "stir":
            record = _emit_stir(step, primary)
        elif action == "postprocess":
            record = _emit_postprocess(step, primary)
        elif action == "redisperse":
            record = _emit_redisperse(step, primary)
        elif action == "prepare_solution":
            record = _emit_solution_prep(step, vessels)
        elif action == "dissolve":
            record = _emit_dissolve(step, primary)
        elif action == "weigh":
            record = _emit_weigh(step, primary)
        elif action == "transfer":
            record = _emit_transfer(step, primary)
        elif action == "add":
            record = _emit_generic_add(step, primary)
        else:
            record = _emit_fallback(step, primary)

        _normalize_reagents_inplace(record)
        if "reagents_structured" not in record:
            _add_structured_reagents_inplace(record)
        emitted.append(record)

        if detail_payload is not None:
            vessels.contents_detailed[primary] = {
                "description": detail_payload["description"],
                "prepared_in_source_step": idx,
                "components": detail_payload["components"],
            }

    return emitted


def build_micro_plan(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    micro_plan: List[Dict[str, Any]] = []
    next_index = 1
    for source_step_index, step in enumerate(steps, start=1):
        for op in step.get("ops", []):
            mp = copy.deepcopy(op)
            verb = mp.pop("op")
            mp["verb"] = verb
            mp["source_step_index"] = source_step_index
            mp["step_index"] = next_index
            next_index += 1
            if mp.get("verb") == "set" and mp.get("param") == "temperature_C" and "unit" not in mp:
                mp["unit"] = "C"
            micro_plan.append(mp)
    return micro_plan


def generate_minimal_plan(doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    micro_plan = doc.get("micro_plan", [])
    micro_plan_min: List[Dict[str, Any]] = []
    timing_delays: List[Dict[str, Any]] = []

    seen_sets: Dict[Tuple[Any, Any, Any], int] = {}
    for op in micro_plan:
        verb = op.get("verb")
        if verb == "set" and op.get("param") == "temperature_C":
            key = (op.get("param"), op.get("value"), op.get("device"))
            if key in seen_sets:
                idx = seen_sets[key]
                entry = micro_plan_min[idx]
                steps_list = entry.setdefault("collapsed_from_steps", [])
                src = op.get("source_step_index")
                if src is not None and src not in steps_list:
                    steps_list.append(src)
                continue

        if verb in {"pick_up", "place", "pour", "set"}:
            entry = copy.deepcopy(op)
            if verb == "set" and op.get("param") == "temperature_C":
                seen_sets[(op.get("param"), op.get("value"), op.get("device"))] = len(micro_plan_min)
            micro_plan_min.append(entry)

        if verb == "wait" and op.get("minutes", 0) > 0:
            timing_delays.append(
                {
                    "step_index": op["step_index"],
                    "verb": "wait",
                    "minutes": op["minutes"],
                }
            )

    return micro_plan_min, timing_delays



# -------- Compatibility exports for legacy tests --------
def detect_oven_dry(line: str) -> Optional[Dict[str, Any]]:
    """Legacy detector kept for test compatibility.

    Returns an oven_dry action for phrases containing oven/dry with optional
    temperature and duration extraction.
    """
    s = strip_tags(_clean_unicode((line or '').strip()))
    if not re.search(r"\b(?:oven|dry(?:ing)?|dried)\b", s, re.I):
        return None
    temp = find_temp_c(s)
    minutes = find_minutes(s)
    return {
        "action": "oven_dry",
        "temperature_C": temp if temp is not None else 80.0,
        "minutes": minutes if minutes is not None else 120.0,
    }


def _legacy_ops_to_micro_ops(step_ops: List[Dict[str, Any]], step_idx: int) -> List[Dict[str, Any]]:
    micro_ops: List[Dict[str, Any]] = []
    for op in step_ops or []:
        item = copy.deepcopy(op)
        op_name = item.pop('op', None)
        if not op_name:
            continue

        if op_name == 'set_hotplate_temperature':
            micro_ops.append({
                'verb': 'set',
                'device': item.get('hotplate_id', 'HP1'),
                'param': 'temperature_C',
                'value': item.get('temperature_C'),
                'unit': 'C',
                'step_index': step_idx,
            })
        elif op_name == 'set_oven_temperature':
            micro_ops.append({
                'verb': 'set',
                'device': item.get('oven_id', 'OV1'),
                'param': 'temperature_C',
                'value': item.get('temperature_C'),
                'unit': 'C',
                'step_index': step_idx,
            })
        elif op_name == 'add_solvent':
            micro_ops.append({
                'verb': 'pour',
                'vessel': item.get('vessel', 'V1'),
                'reagent': item.get('solvent') or item.get('reagent'),
                'volume': item.get('volume'),
                'volume_units': item.get('volume_units', 'mL'),
                'step_index': step_idx,
            })
        elif op_name == 'move_to_oven':
            micro_ops.append({
                'verb': 'pick_up',
                'vessel': item.get('tube') or item.get('vessel', 'V1'),
                'step_index': step_idx,
            })
            micro_ops.append({
                'verb': 'place',
                'vessel': item.get('tube') or item.get('vessel', 'V1'),
                'to': item.get('oven_id', 'OV1'),
                'step_index': step_idx,
            })
        else:
            item['verb'] = op_name
            item.setdefault('step_index', step_idx)
            micro_ops.append(item)
    return micro_ops


def apply_postprocessing(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility normalizer used by legacy tests.

    It performs a narrow normalization pass:
      * rebuild micro_plan from steps[*].ops when absent
      * canonicalize device aliases like ``water bath`` -> ``HP1``
      * default temperature set units to ``C``
      * default pour volume units from add_solvent-derived ops to ``mL``
      * insert ``pick_up`` and ``place`` immediately before hotplate set ops
      * attach ``_executor`` metadata with repairs list
    """
    result = copy.deepcopy(doc or {})
    repairs: List[str] = []

    device_aliases = {
        'water bath': 'HP1',
        'hot plate': 'HP1',
        'hotplate': 'HP1',
        'heating plate': 'HP1',
        'stir plate': 'SP1',
        'stirrer': 'SP1',
        'centrifuge': 'CF1',
        'oven': 'OV1',
    }

    micro_plan = copy.deepcopy(result.get('micro_plan') or [])
    if not micro_plan:
        for step_idx, step in enumerate(result.get('steps', []), start=1):
            if step.get('micro_ops'):
                ops = copy.deepcopy(step['micro_ops'])
                for op in ops:
                    op.setdefault('step_index', step_idx)
                micro_plan.extend(ops)
            elif step.get('ops'):
                micro_plan.extend(_legacy_ops_to_micro_ops(step.get('ops', []), step_idx))
        if micro_plan:
            repairs.append('rebuilt_micro_plan_from_steps')

    # canonicalize and fill defaults
    for op in micro_plan:
        device = op.get('device')
        if isinstance(device, str):
            mapped = device_aliases.get(device.lower().strip())
            if mapped and mapped != device:
                op['device'] = mapped
                repairs.append(f'canonicalized_device_{device.lower().replace(" ", "_")}_to_{mapped}')
        if op.get('verb') == 'set' and op.get('param') == 'temperature_C':
            op.setdefault('unit', 'C')
        if op.get('verb') == 'pour' and op.get('volume') is not None:
            op.setdefault('volume_units', 'mL')

    # insert hotplate pick_up/place immediately before set operations
    enhanced: List[Dict[str, Any]] = []
    for op in micro_plan:
        if op.get('verb') == 'set' and op.get('device') == 'HP1' and op.get('param') == 'temperature_C':
            vessel = op.get('vessel', 'V1')
            need_insert = True
            if len(enhanced) >= 2:
                prev2, prev1 = enhanced[-2], enhanced[-1]
                need_insert = not (
                    prev2.get('verb') == 'pick_up' and prev2.get('vessel') == vessel and
                    prev1.get('verb') == 'place' and prev1.get('vessel') == vessel and prev1.get('to') == 'HP1'
                )
            if need_insert:
                enhanced.append({'verb': 'pick_up', 'vessel': vessel, 'step_index': op.get('step_index', 1)})
                enhanced.append({'verb': 'place', 'vessel': vessel, 'to': 'HP1', 'step_index': op.get('step_index', 1)})
                repairs.append('inserted_hotplate_pickup_place')
        enhanced.append(op)

    # ensure deterministic unique step indexes
    next_idx = 1
    for op in enhanced:
        if 'source_step_index' not in op and 'step_index' in op:
            op['source_step_index'] = op['step_index']
        op['step_index'] = next_idx
        next_idx += 1

    result['micro_plan'] = enhanced
    result.setdefault('_executor', {})
    result['_executor']['schema_version'] = 'executor.v1'
    result['_executor']['repairs'] = repairs
    result['_executor']['postprocessing_applied'] = True
    return result

def validate_execution_plan(plan: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for step in plan.get("micro_plan", []):
        if step.get("verb") == "set":
            device = step.get("device")
            param = step.get("param")
            value = step.get("value")
            if device == "HP1" and param == "temperature_C":
                if value > 200:
                    errors.append(f"Temperature {value}°C exceeds safety limit of 200°C")
                if value < -20:
                    errors.append(f"Temperature {value}°C below safety limit of -20°C")
            if device == "SP1" and param == "rpm":
                if value > 2000:
                    errors.append(f"Stir speed {value} rpm exceeds equipment limit of 2000 rpm")
                if value < 0:
                    errors.append(f"Stir speed {value} rpm cannot be negative")
            if device == "OV1" and param == "temperature_C" and value > 300:
                errors.append(f"Oven temperature {value}°C exceeds safety limit of 300°C")

    available_devices = set(plan.get("devices", {}).values())
    for step in plan.get("micro_plan", []):
        if step.get("device") and step["device"] not in available_devices:
            errors.append(f"Required device {step['device']} not available in device registry")
    return errors


def convert_text_to_robot_ops(text: str) -> Dict[str, Any]:
    hardware = parse_hardware(text)
    vessels = VesselRegistry(hardware)
    primary_vessel = vessels.ensure_glassware("Beaker")
    semantic_steps, _context = semantic_parse(text, vessels)
    records = emit_steps(semantic_steps, vessels)

    result: Dict[str, Any] = {
        "_executor": {
            "schema_version": "executor.v1",
            "converter": "converter_v2",
            "postprocessing_applied": False,
            "repairs": [],
        },
        "devices": {
            "centrifuge_id": DEVICE_IDS["centrifuge_id"],
            "stir_plate_id": DEVICE_IDS["stir_plate_id"],
        },
        "hardware": hardware,
        "vessel_registry": vessels.as_dict() or {primary_vessel: "Beaker"},
        "vessel_contents_detailed": vessels.contents_detailed,
        "steps": records,
    }

    if "V1" not in result["vessel_registry"]:
        result["vessel_registry"]["V1"] = "Beaker"
    if any(step.get("action") == "postprocess" for step in records):
        result["vessel_registry"].setdefault("V1_tube", "Centrifuge Tube")

    micro_plan = build_micro_plan(records)
    result["micro_plan"] = micro_plan
    micro_plan_min, timing_delays = generate_minimal_plan(result)
    result["micro_plan_min"] = micro_plan_min
    result["timing_delays"] = timing_delays

    errors = validate_execution_plan(result)
    if errors:
        result["_executor"]["validation_errors"] = errors

    return result


def validate_step(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("input must be a string")
    raw = text.strip()
    if not raw:
        raise ValueError("input text is empty")
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        raise ValueError("JSON input must be an object")
    except json.JSONDecodeError:
        pass
    data: Dict[str, Any] = {}
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"line {lineno}: missing ':' separator")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"line {lineno}: key is empty")
        data[key] = value
    if not data:
        raise ValueError("no key:value pairs found")
    return data


def validate_file(path: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists():
        raise ValueError(f"file '{path}' does not exist")
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                item = validate_step(line)
            except ValueError as exc:
                raise ValueError(f"{p.name}:{lineno}: {exc}") from None
            items.append(item)
    return items


if __name__ == "__main__":
    import argparse

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
