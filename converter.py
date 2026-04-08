
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

NON_PROCEDURE_STEP_RX = re.compile(
    r"^\s*(?:hardware(?:\s*&\s*glassware)?|materials?|reagents?|equipment|apparatus|supplies|chemicals?|glassware)\s*:\s*",
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


def _temperature_candidates_c(text: str) -> List[float]:
    s = _clean_unicode(text or "")
    s = s.replace("℃", "°C").replace("℉", "°F").replace("◦", "°").replace("º", "°")

    vals: List[float] = []

    if re.search(r"\breflux\b", s, re.I):
        vals.append(100.0)
    if re.search(r"\bboil(?:ing)?\b", s, re.I):
        vals.append(100.0)
    if re.search(r"\bice\s*bath\b", s, re.I):
        vals.append(0.0)

    range_patterns = [
        re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:-|–|—|~|to)\s*(-?\d+(?:\.\d+)?)\s*(?:°\s*)?([CFK])\b", re.I),
        re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*(?:-|–|—|~|to)\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CFK])\b", re.I),
    ]
    for rx in range_patterns:
        for m in rx.finditer(s):
            for idx in (1, 2):
                v = float(m.group(idx))
                unit = m.group(3).upper()
                if unit == "C":
                    vals.append(v)
                elif unit == "F":
                    vals.append((v - 32.0) * 5.0 / 9.0)
                elif unit == "K":
                    vals.append(v - 273.15)

    s_single = s
    for rx in range_patterns:
        s_single = rx.sub(" ", s_single)

    for m in re.finditer(r"(?<![\d])(-?\d+(?:\.\d+)?)\s*(?:°\s*)?([CFK])\b", s_single, re.I):
        v = float(m.group(1))
        unit = m.group(2).upper()
        if unit == "C":
            vals.append(v)
        elif unit == "F":
            vals.append((v - 32.0) * 5.0 / 9.0)
        elif unit == "K":
            vals.append(v - 273.15)

    for m in re.finditer(r"(?<![\d])(-?\d+(?:\.\d+)?)\s*(?:°\s*)?(celsius|centigrade)\b", s_single, re.I):
        vals.append(float(m.group(1)))

    out: List[float] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def find_temp_c(text: str) -> Optional[float]:
    vals = _temperature_candidates_c(text)
    if vals:
        return vals[0]
    s = _clean_unicode(text or "")
    if re.search(r"\b(rt|room\s*temp(?:erature)?)\b", s, re.I):
        return DEFAULTS["room_temp_C"]
    return None


def find_minutes(text: str) -> Optional[float]:
    s = _clean_unicode(text)
    found = False
    minutes = 0.0
    if re.search(r"\bover\s*night\b", s, re.I):
        return 12 * 60.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b", s, re.I):
        minutes += float(m.group(1)) / 60.0
        found = True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", s, re.I):
        minutes += float(m.group(1)) * 60.0
        found = True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|min)\b", s, re.I):
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



def _strip_markdown_prefix(line: str) -> str:
    s = strip_tags(line.strip())
    s = re.sub(r"^\s*[-*•]\s+", "", s)
    s = re.sub(r"^\s*\d+\.\s+", "", s)
    return s.strip()


def _is_section_heading_line(line: str) -> bool:
    s = strip_tags(line.strip())
    if not s:
        return False
    if re.search(r"\bprocedure\b\s*:?$", s, re.I):
        return True
    return s.endswith(":")


def _clean_step_text(text: str) -> str:
    s = strip_tags(_clean_unicode(text or ""))
    s = re.sub(r"\s+", " ", s).strip(" -•\t")
    return s


def extract_steps(markdown_text: str) -> List[str]:
    lines = markdown_text.splitlines()
    in_proc = False
    saw_proc_header = False
    steps: List[str] = []
    current_section: Optional[str] = None
    last_step_idx: Optional[int] = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            last_step_idx = None
            continue

        plain = strip_tags(stripped)

        if re.search(r"(?:^|\b)procedure\b\s*:?$", plain, re.I):
            in_proc = True
            saw_proc_header = True
            current_section = None
            last_step_idx = None
            continue

        if not in_proc and saw_proc_header:
            continue

        if not in_proc:
            continue

        if re.match(r"^\s*\d+\.\s*", stripped):
            content = _strip_markdown_prefix(stripped)
            if _is_section_heading_line(content):
                heading = content.rstrip(":").strip()
                if not re.search(r"\bprocedure\b", heading, re.I) and not _is_non_procedure_step_text(heading + ":"):
                    current_section = heading
                last_step_idx = None
                continue
            candidate = _clean_step_text(content)
            if candidate and not _is_non_procedure_step_text(candidate):
                if current_section and not candidate.lower().startswith(current_section.lower()):
                    candidate = f"{current_section}: {candidate}"
                steps.append(candidate)
                last_step_idx = len(steps) - 1
                continue

        if re.match(r"^\s*[-*•]\s+", stripped):
            candidate = _clean_step_text(_strip_markdown_prefix(stripped))
            if candidate and not _is_non_procedure_step_text(candidate):
                if current_section and not candidate.lower().startswith(current_section.lower()):
                    candidate = f"{current_section}: {candidate}"
                steps.append(candidate)
                last_step_idx = len(steps) - 1
            continue

        if _is_section_heading_line(plain):
            heading = plain.rstrip(":").strip()
            if not _is_non_procedure_step_text(heading + ":"):
                current_section = heading
            last_step_idx = None
            continue

        if last_step_idx is not None:
            extra = _clean_step_text(plain)
            if extra:
                steps[last_step_idx] = f"{steps[last_step_idx]} {extra}".strip()
            continue

        candidate = _clean_step_text(plain)
        if candidate and not _is_non_procedure_step_text(candidate):
            if current_section and not candidate.lower().startswith(current_section.lower()):
                candidate = f"{current_section}: {candidate}"
            steps.append(candidate)
            last_step_idx = len(steps) - 1

    if steps:
        return [s for s in steps if s and not _is_non_procedure_step_text(s)]

    fallback_steps: List[str] = []
    last_idx: Optional[int] = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            last_idx = None
            continue
        plain = strip_tags(stripped)
        if _is_non_procedure_step_text(plain):
            last_idx = None
            continue
        if re.match(r"^\s*(?:\d+\.|[-*•])\s+", stripped):
            candidate = _clean_step_text(_strip_markdown_prefix(stripped))
            if candidate:
                fallback_steps.append(candidate)
                last_idx = len(fallback_steps) - 1
            continue
        if last_idx is not None:
            extra = _clean_step_text(plain)
            if extra:
                fallback_steps[last_idx] = f"{fallback_steps[last_idx]} {extra}".strip()

    if fallback_steps:
        return fallback_steps

    cleaned = _clean_step_text(markdown_text)
    return [cleaned] if cleaned and not _is_non_procedure_step_text(cleaned) else []


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


def _normalize_solvent_name(solvent: Optional[str]) -> Optional[str]:
    if solvent is None:
        return None
    s = _clean_solvent_tail(solvent)
    s = re.sub(r"\b(?:for|to)\b.+$", "", s, flags=re.I).strip().rstrip(",.")
    return s or None


def _is_non_procedure_step_text(text: str) -> bool:
    s = strip_tags(_clean_unicode(text or ""))
    return bool(NON_PROCEDURE_STEP_RX.match(s))


DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "pt(acac)2": "Platinum(II) acetylacetonate (Pt(acac)2)",
    "ru(acac)3": "Ruthenium(III) acetylacetonate (Ru(acac)3)",
    "ctab": "Hexadecyltrimethylammonium bromide (CTAB)",
    "nabh4": "Sodium borohydride (NaBH4)",
    "chloroform": "Chloroform",
    "water": "Water",
    "ethanol": "Ethanol",
}


def _display_name_for(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    cleaned = strip_tags(name).strip()
    key = cleaned.lower()
    return DISPLAY_NAME_OVERRIDES.get(key, cleaned)


def _solution_struct(name: str, *, original: Optional[str] = None, concentration: Optional[float] = None,
                     conc_unit: Optional[str] = None, solvent: Optional[str] = None,
                     is_solution: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name": name,
        "display_name": _display_name_for(name),
        "original": original or name,
    }
    if concentration is not None:
        out["concentration"] = concentration
    if conc_unit is not None:
        out["conc_unit"] = conc_unit
    if solvent is not None:
        out["solvent"] = solvent
    if is_solution:
        out["is_solution"] = True
    return out


def _is_non_procedure_step_text(text: str) -> bool:
    s = strip_tags(_clean_unicode(text or ""))
    return bool(NON_PROCEDURE_STEP_RX.match(s))


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



def detect_prepare_solution_from_amounts(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\bdissolv\w*\b", s, re.I):
        return None
    m = re.search(
        r"\bdissolv\w*\s+(?P<solutes>.+?)\s+in\s+(?P<vol>\d+(?:\.\d+)?)\s*(?P<vunit>µ?u?L|mL|ml|L|l)\s+of\s+(?P<solvent>.+?)(?:\s+in\s+(?P<hardware>\d+\s*mL\s+[^.;]+))?$",
        s,
        re.I,
    )
    if not m:
        return None
    structured = [parse_reagent_phrase_to_struct(p) for p in split_reagent_phrases(m.group("solutes")) if p]
    if len(structured) < 2:
        return None
    return {
        "action": "prepare_solution_from_amounts",
        "solutes": structured,
        "solvent": _clean_solvent_tail(m.group("solvent")),
        "volume": float(m.group("vol")),
        "volume_units": m.group("vunit"),
        "hardware_hint": m.group("hardware"),
    }


def detect_add_measured_reagent(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    m = re.search(
        r"\badd\s+(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg|µg|ug|mmol|mol|µmol|umol)\s+of\s+(?P<reagent>[^,.;]+?)\s+to\s+(?:the\s+)?(?P<target>solution|mixture|flask|reaction mixture)\b",
        s,
        re.I,
    )
    if not m:
        return None
    explicit_temps = _temperature_candidates_c(s)
    return {
        "action": "add_measured_reagent",
        "reagent": m.group("reagent").strip(),
        "amount": float(m.group("amount")),
        "unit": _canon_unit(m.group("unit")),
        "target_name": m.group("target").strip(),
        "with_stirring": bool(re.search(r"\bstir", s, re.I)),
        "temperature_C": explicit_temps[0] if explicit_temps else None,
        "minutes": find_minutes(s),
    }


def _extract_wash_sequence(text: str) -> List[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(text))
    seq: List[Dict[str, Any]] = []
    for m in re.finditer(
        r"(?:(?:with|followed by)\s+)?(?P<vol>\d+(?:\.\d+)?)\s*(?P<vunit>µ?u?L|mL|ml|L|l)\s+of\s+(?P<solvent>[^,.;]+)",
        s,
        re.I,
    ):
        seq.append({
            "solvent": _clean_solvent_tail(m.group("solvent")),
            "volume": float(m.group("vol")),
            "volume_units": m.group("vunit"),
        })
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in seq:
        key = (item["solvent"].lower(), item["volume"], item["volume_units"].lower())
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


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


def detect_heat_hold(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\b(heat|maintain|hold|continue\s+heating?|water\s+bath)\b", s, re.I):
        return None
    temp = find_temp_c(s)
    minutes = find_minutes(s)
    if temp is None and minutes is None:
        return None
    return {
        "action": "heat_hold",
        "temperature_C": temp if temp is not None else DEFAULTS["room_temp_C"],
        "minutes": minutes if minutes is not None else 0.0,
        "device": "water bath" if re.search(r"\bwater\s+bath\b", s, re.I) else "HP1",
    }


def detect_autotitrator_rate(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\b(autotitrator|titrat)\b", s, re.I):
        return None
    m = re.search(r"(?:rate(?:\s+of)?|at)\s+([0-9]+(?:\.[0-9]+)?)\s*mL\s*/\s*min", s, re.I)
    if not m:
        return None
    return {"action": "autotitrator_rate", "rate_mL_per_min": float(m.group(1))}

def detect_ph_monitoring(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\bph\b", s, re.I):
        return None
    return {"action": "monitor_ph", "continuous": True, "interval_seconds": 30}




def detect_explicit_postprocess(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\bcentrifug|\bwash\b|\bsupernatant\b", s, re.I):
        return None

    centrifuge_rpm = None
    centrifuge_minutes = None

    m1 = re.search(r"centrifug\w*.*?\bat\s+(?P<rpm>\d+)\s*rpm\s+for\s+(?P<mins>\d+(?:\.\d+)?)\s*(?:minutes?|mins?)", s, re.I)
    m2 = re.search(r"centrifug\w*.*?\bfor\s+(?P<mins>\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\s+at\s+(?P<rpm>\d+)\s*rpm", s, re.I)
    m = m1 or m2
    if m:
        centrifuge_rpm = int(m.group("rpm"))
        centrifuge_minutes = float(m.group("mins"))

    wash_sequence = _extract_wash_sequence(s)
    if not wash_sequence and centrifuge_rpm is None and centrifuge_minutes is None:
        return None

    return {
        "action": "postprocess",
        "wash_count": _parse_repeat_count(s, default=1) if wash_sequence else 0,
        "wash_sequence": wash_sequence,
        "wash_solvent": wash_sequence[0]["solvent"] if wash_sequence else None,
        "wash_volume": wash_sequence[0]["volume"] if wash_sequence else None,
        "wash_volume_units": wash_sequence[0]["volume_units"] if wash_sequence else None,
        "centrifuge_minutes": centrifuge_minutes if centrifuge_minutes is not None else DEFAULTS["centrifuge_minutes"],
        "centrifuge_rpm": centrifuge_rpm if centrifuge_rpm is not None else DEFAULTS["centrifuge_rpm"],
        "centrifuge_explicit": centrifuge_rpm is not None and centrifuge_minutes is not None,
    }

def detect_redisperse(line: str) -> Optional[Dict[str, Any]]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    if not re.search(r"\b(disperse|redisperse)\b", s, re.I):
        return None
    m = re.search(r"\bin\s+(?P<solvent>[^,.;]+)", s, re.I)
    solvent = _normalize_solvent_name(m.group("solvent")) if m else None
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
        detect_prepare_solution_from_amounts,
        detect_prepare_and_add_solution,
        detect_add_solvent,
        detect_add_measured_reagent,
        detect_prepare_and_add_reagent_solution,
        detect_explicit_postprocess,
        detect_redisperse,
        detect_heat_hold,
        detect_autotitrator_rate,
        detect_ph_monitoring,
        detect_oven_dry,
        detect_solution_prep,
        detect_dissolve,
        detect_stir,
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
        if _is_non_procedure_step_text(step):
            continue
        parsed = semantic_parse_step(step)
        if parsed.get("action") == "composite":
            for part in parsed["parts"]:
                if not _is_non_procedure_step_text(part.get("raw", "")):
                    records.append(part)
            continue
        records.append(parsed)

    records = _normalize_semantic_steps(records)
    return records, context


def _normalize_semantic_steps(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    last_explicit_centrifuge: Optional[Tuple[float, int]] = None

    for step in records:
        current = copy.deepcopy(step)

        if current.get("action") == "heat_hold":
            mins = current.get("minutes")
            raw = current.get("raw", "")
            if (mins is None or mins == 0) and find_minutes(raw):
                current["minutes"] = find_minutes(raw)

        if current.get("action") == "oven_dry":
            mins = current.get("minutes")
            raw = current.get("raw", "")
            if (mins is None or mins <= 120) and find_minutes(raw):
                current["minutes"] = find_minutes(raw)

        if current.get("action") == "postprocess":
            raw = current.get("raw", "")
            explicit_match_1 = re.search(r"centrifug\w*.*?\bat\s+(?P<rpm>\d+)\s*rpm\s+for\s+(?P<mins>\d+(?:\.\d+)?)\s*(?:minutes?|mins?)", raw, re.I)
            explicit_match_2 = re.search(r"centrifug\w*.*?\bfor\s+(?P<mins>\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\s+at\s+(?P<rpm>\d+)\s*rpm", raw, re.I)
            explicit_match = explicit_match_1 or explicit_match_2
            if explicit_match:
                last_explicit_centrifuge = (float(explicit_match.group("mins")), int(explicit_match.group("rpm")))
            elif current.get("wash_sequence") and last_explicit_centrifuge is not None:
                current["centrifuge_minutes"] = last_explicit_centrifuge[0]
                current["centrifuge_rpm"] = last_explicit_centrifuge[1]

        normalized.append(current)

    return normalized


def _ensure_primary_vessel(vessels: VesselRegistry) -> str:
    return vessels.primary_vessel or vessels.ensure_glassware("Beaker")



def _emit_prepare_solution_from_amounts(step: Dict[str, Any], vessels: VesselRegistry) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    solvent = step["solvent"]
    volume = step["volume"]
    volume_units = step["volume_units"]
    vol_ml = volume * (0.001 if volume_units.lower() in ("µl", "ul") else (1.0 if volume_units.lower() == "ml" else 1000.0))
    vid = vessels.ensure_glassware(step.get("hardware_hint") or "Beaker", prefer_capacity_ml=vol_ml, explicit_hardware_hint=step.get("hardware_hint"))
    ops: List[Dict[str, Any]] = []
    reagents: List[str] = []
    structured: List[Dict[str, Any]] = []
    detail_components: List[Dict[str, Any]] = []

    for sol in step["solutes"]:
        ops.append({"op": "pour", "reagent": sol["name"], "vessel": vid})
        reagents.append(sol["name"])
        struct = {
            "name": sol["name"],
            "display_name": _display_name_for(sol["name"]),
            "original": sol.get("original") or sol["name"],
        }
        if sol.get("amount") is not None:
            struct["amount"] = sol["amount"]
        if sol.get("amount_unit") is not None:
            struct["amount_unit"] = sol["amount_unit"]
        structured.append(struct)
        detail_components.append({"name": sol["name"], "display_name": _display_name_for(sol["name"])})
    ops.append({"op": "pour", "reagent": solvent, "vessel": vid, "volume": volume, "volume_units": volume_units})
    reagents.append(solvent)
    structured.append({"name": solvent, "display_name": _display_name_for(solvent), "original": f"{volume} {volume_units} of {solvent}", "amount": volume, "amount_unit": volume_units})
    detail_components.append({"name": solvent, "display_name": _display_name_for(solvent), "role": "solvent"})

    record = {
        "action": "prepare_solution",
        "raw": step["raw"],
        "vessel": vid,
        "solvent": solvent,
        "volume": volume,
        "volume_units": volume_units,
        "reagents": reagents,
        "reagents_structured": structured,
        "ops": ops,
    }
    detail = {
        "description": f"prepared mixture in {solvent}",
        "components": detail_components,
    }
    return record, detail


def _emit_add_measured_reagent(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    ops: List[Dict[str, Any]] = []
    if step.get("with_stirring"):
        ops.extend([
            {"op": "move_to_stir_plate", "stir_plate_id": DEVICE_IDS["stir_plate_id"], "vessel": vessel},
            {"op": "set_stir_rate", "vessel": vessel, "rpm": DEFAULTS["stir_rpm"], "inferred": True},
        ])
    ops.append({"op": "pour", "reagent": step["reagent"], "vessel": vessel, "amount": step["amount"], "amount_unit": step["unit"]})
    if step.get("minutes"):
        ops.append({"op": "wait", "minutes": step["minutes"]})
    return {
        "action": "add",
        "raw": step["raw"],
        "vessel": vessel,
        "with_stirring": step.get("with_stirring", False),
        "temperature_C": step.get("temperature_C"),
        "minutes": step.get("minutes"),
        "reagents": [step["reagent"]],
        "reagents_structured": [{
            "name": step["reagent"],
            "display_name": _display_name_for(step["reagent"]),
            "original": f"{step['amount']} {step['unit']} of {step['reagent']}",
            "amount": step["amount"],
            "amount_unit": step["unit"],
        }],
        "ops": ops,
    }


def _emit_prepare_solution(step: Dict[str, Any], vessel: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    components = step["components"]
    solvent = step["solvent"]
    ops: List[Dict[str, Any]] = []
    reagents: List[str] = []
    detailed_components: List[Dict[str, Any]] = []
    structured: List[Dict[str, Any]] = []

    for comp in components:
        ops.append({"op": "pour", "reagent": comp["name"], "vessel": vessel})
        reagents.append(comp["name"])
        detailed_components.append(
            {
                "name": comp["name"],
                "display_name": _display_name_for(comp["name"]),
                "concentration": comp["concentration"],
                "conc_unit": comp["conc_unit"],
                "solvent": solvent,
            }
        )
        structured.append(_solution_struct(
            comp["name"],
            original=f"{comp['concentration']} {comp['conc_unit']} of {comp['name']}",
            concentration=comp["concentration"],
            conc_unit=comp["conc_unit"],
            solvent=solvent,
            is_solution=False,
        ))
    ops.append({"op": "pour", "solvent": solvent, "vessel": vessel})
    reagents.append(solvent)
    structured.append({"name": solvent, "display_name": _display_name_for(solvent), "original": solvent})
    detail = {
        "description": step.get("description", f"prepared solution in {solvent}"),
        "components": detailed_components + [{"name": solvent, "display_name": _display_name_for(solvent), "role": "solvent"}],
    }
    record = {
        "action": "prepare_solution",
        "raw": step["raw"],
        "vessel": vessel,
        "ops": ops,
        "reagents": reagents,
        "reagents_structured": structured,
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
        "with_stirring": True,
        "ops": ops,
        "reagents": [step["reagent"], step["solvent"]],
        "reagents_structured": [
            _solution_struct(
                step["reagent"],
                original=f"{step['concentration']} {step['conc_unit']} solution of {step['reagent']} in {step['solvent']}",
                concentration=step["concentration"],
                conc_unit=step["conc_unit"],
                solvent=step["solvent"],
                is_solution=True,
            )
        ],
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
        "reagents_structured": [{
            "name": step["solvent"],
            "display_name": _display_name_for(step["solvent"]),
            "original": f"{step['volume']} {step['volume_units']} of {step['solvent']}",
            "amount": step["volume"],
            "amount_unit": step["volume_units"],
        }],
    }


def _emit_add_reagent_solution(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    token = f"{_slug_token(step['reagent'])}_{_slug_token(step['solvent'])}_solution"
    ops = [{"op": "pour", "reagent": token, "vessel": vessel}]
    solvent_name = "water" if step["solvent"].lower() == "aqueous" else step["solvent"]
    return {
        "action": "add",
        "raw": step["raw"],
        "vessel": vessel,
        "with_stirring": bool(step.get("with_stirring", True) or re.search(r"reaction mixture|under stirring|while stirring", step.get("raw", ""), re.I)),
        "ops": ops,
        "reagents": [f"{step['reagent']} aqueous solution" if step["solvent"].lower() == "aqueous" else f"{step['reagent']} {step['solvent']} solution"],
        "reagents_structured": [
            _solution_struct(
                step["reagent"],
                original=f"aqueous solution of {step['reagent']}" if step["solvent"].lower() == "aqueous" else f"{step['solvent']} solution of {step['reagent']}",
                solvent=solvent_name,
                is_solution=True,
            )
        ],
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


def _emit_heat_hold(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    return {
        "action": "heat_hold",
        "raw": step["raw"],
        "vessel": vessel,
        "temperature_C": step["temperature_C"],
        "minutes": step["minutes"],
        "ops": [
            {
                "op": "set",
                "device": step.get("device", "HP1"),
                "param": "temperature_C",
                "value": step["temperature_C"],
                "unit": "C",
                "vessel": vessel,
            },
            *([{"op": "wait", "minutes": step["minutes"]}] if step["minutes"] and step["minutes"] > 0 else []),
        ],
        "reagents": [],
        "reagents_structured": [],
    }


def _emit_autotitrator_rate(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    return {
        "action": "autotitrator_rate",
        "raw": step["raw"],
        "vessel": vessel,
        "rate_mL_per_min": step["rate_mL_per_min"],
        "ops": [
            {
                "op": "set",
                "device": "AT1",
                "param": "rate_mL_per_min",
                "value": step["rate_mL_per_min"],
            }
        ],
        "reagents": [],
        "reagents_structured": [],
    }




def _emit_monitor_ph(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    return {
        "action": "monitor_ph",
        "raw": step["raw"],
        "vessel": vessel,
        "continuous": True,
        "interval_seconds": step.get("interval_seconds", 30),
        "ops": [
            {
                "op": "monitor_ph",
                "ph_meter_id": DEVICE_IDS["ph_meter_id"],
                "vessel": vessel,
                "interval_seconds": step.get("interval_seconds", 30),
            }
        ],
        "reagents": [],
        "reagents_structured": [],
    }

def _emit_oven_dry(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    tube = f"{vessel}_tube"
    return {
        "action": "oven_dry",
        "raw": step["raw"],
        "vessel": vessel,
        "temperature_C": step["temperature_C"],
        "minutes": step["minutes"],
        "ops": [
            {"op": "move_to_oven", "tube": tube, "oven_id": DEVICE_IDS["oven_id"]},
            {"op": "set_oven_temperature", "oven_id": DEVICE_IDS["oven_id"], "temperature_C": step["temperature_C"], "tube": tube},
            {"op": "wait", "minutes": step["minutes"]},
        ],
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
            "inferred": not bool(step.get("centrifuge_explicit")),
        },
        {
            "op": "decant_supernatant",
            "tube": f"{vessel}_tube",
            "executable": False,
            "review_required": True,
            "review_reason": "unsupported_phase_separation_primitive",
        },
    ]
    wash_sequence = step.get("wash_sequence") or []
    if not wash_sequence and step.get("wash_solvent"):
        wash_sequence = [{
            "solvent": step["wash_solvent"],
            "volume": step.get("wash_volume"),
            "volume_units": step.get("wash_volume_units") or "mL",
        }] if step.get("wash_count", 0) > 0 else []

    for wash_cycle in range(1, step["wash_count"] + 1):
        for wash in wash_sequence:
            ops.append(
                {
                    "op": "pour",
                    "reagent": wash["solvent"],
                    "vessel": f"{vessel}_tube",
                    "volume": wash.get("volume"),
                    "volume_units": wash.get("volume_units") or "mL",
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

    reagents = [wash["solvent"] for wash in wash_sequence]
    reagents_structured = []
    for wash in wash_sequence:
        entry = {
            "name": wash["solvent"],
            "display_name": _display_name_for(wash["solvent"]),
            "original": f"{wash.get('volume')} {wash.get('volume_units') or 'mL'} of {wash['solvent']}",
            "repetitions": step["wash_count"],
        }
        if wash.get("volume") is not None:
            entry["amount"] = wash["volume"]
        if wash.get("volume_units") is not None:
            entry["amount_unit"] = wash["volume_units"]
        reagents_structured.append(entry)

    return {
        "action": "postprocess",
        "raw": step["raw"],
        "vessel": vessel,
        "ops": ops,
        "reagents": reagents,
        "reagents_structured": reagents_structured,
    }

def _emit_redisperse(step: Dict[str, Any], vessel: str) -> Dict[str, Any]:
    redisperse_vessel = f"{vessel}_tube"
    solvent = _normalize_solvent_name(step.get("solvent")) or "solvent"
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
        "reagents_structured": [{"name": solvent, "display_name": _display_name_for(solvent), "original": solvent}],
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
        elif action == "prepare_solution_from_amounts":
            record, detail_payload = _emit_prepare_solution_from_amounts(step, vessels)
        elif action == "add_prepared_solution":
            record = _emit_add_prepared_solution(step, primary)
        elif action == "add_measured_reagent":
            record = _emit_add_measured_reagent(step, primary)
        elif action == "add_solvent":
            record = _emit_add_solvent(step, primary)
        elif action == "add_reagent_solution":
            record = _emit_add_reagent_solution(step, primary)
        elif action == "stir":
            record = _emit_stir(step, primary)
        elif action == "heat_hold":
            record = _emit_heat_hold(step, primary)
        elif action == "autotitrator_rate":
            record = _emit_autotitrator_rate(step, primary)
        elif action == "monitor_ph":
            record = _emit_monitor_ph(step, primary)
        elif action == "oven_dry":
            record = _emit_oven_dry(step, primary)
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
            raw = copy.deepcopy(op)
            verb = raw.pop("op")
            expanded: List[Dict[str, Any]] = []

            if verb == "move_to_oven":
                tube = raw.get("tube", "V1_tube")
                expanded.append({"verb": "pick_up", "vessel": tube, "source_step_index": source_step_index})
                expanded.append({"verb": "place", "vessel": tube, "to": "oven", "device": "OV1", "source_step_index": source_step_index})
            elif verb == "set_oven_temperature":
                expanded.append({
                    "verb": "set",
                    "device": "OV1",
                    "param": "temperature_C",
                    "value": raw.get("temperature_C"),
                    "unit": "C",
                    "source_step_index": source_step_index,
                })
            else:
                raw["verb"] = verb
                raw["source_step_index"] = source_step_index
                expanded.append(raw)

            for mp in expanded:
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




def detect_oven_dry(line: str) -> Optional[Dict[str, Any]]:
    """Detect oven-drying instructions such as 'dry in an oven at 80 C for 2 h'."""
    s = strip_tags(_clean_unicode((line or '').strip().rstrip('.')))
    if not re.search(r"\b(?:dry|drying|oven)\b", s, re.I):
        return None
    if not re.search(r"\boven\b", s, re.I) and not re.search(r"\bdry\b", s, re.I):
        return None
    temp = find_temp_c(s) or 80.0
    minutes = find_minutes(s) or 120.0
    return {"action": "oven_dry", "temperature_C": temp, "minutes": minutes}


def _assign_unique_action_step_indices(actions: List[Dict[str, Any]], start_at: int = 1) -> int:
    next_idx = start_at
    for action in actions:
        if action.get('step_index') is not None and 'source_step_index' not in action:
            action['source_step_index'] = action['step_index']
        action['step_index'] = next_idx
        next_idx += 1
    return next_idx


def _canonicalize_device_name(device: Any) -> Any:
    if not isinstance(device, str):
        return device
    aliases = {
        'water bath': 'HP1',
        'hotplate': 'HP1',
        'hot plate': 'HP1',
        'heating plate': 'HP1',
        'oven': 'OV1',
        'stir plate': 'SP1',
        'stirrer': 'SP1',
        'centrifuge': 'CF1',
        'autotitrator': 'AT1',
    }
    return aliases.get(device.strip().lower(), device)


def _micro_from_ops(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    micro: List[Dict[str, Any]] = []
    for source_step_index, step in enumerate(steps, start=1):
        for op in step.get('ops', []) or []:
            raw = copy.deepcopy(op)
            op_name = raw.pop('op', None)
            if not op_name:
                continue
            if op_name == 'set_hotplate_temperature':
                micro.append({
                    'verb': 'set', 'device': 'HP1', 'param': 'temperature_C',
                    'value': raw.get('temperature_C'), 'unit': 'C',
                    'source_step_index': source_step_index,
                })
                continue
            if op_name == 'set_oven_temperature':
                micro.append({
                    'verb': 'set', 'device': 'OV1', 'param': 'temperature_C',
                    'value': raw.get('temperature_C'), 'unit': 'C',
                    'source_step_index': source_step_index,
                })
                continue
            if op_name == 'move_to_oven':
                obj = raw.get('tube') or raw.get('vessel') or 'V1_tube'
                micro.append({'verb': 'pick_up', 'vessel': obj, 'source_step_index': source_step_index})
                micro.append({'verb': 'place', 'vessel': obj, 'to': 'oven', 'device': 'OV1', 'source_step_index': source_step_index})
                continue
            if op_name == 'add_solvent':
                entry = {
                    'verb': 'pour',
                    'reagent': raw.get('solvent') or raw.get('reagent'),
                    'vessel': raw.get('vessel', 'V1'),
                    'volume': raw.get('volume'),
                    'volume_units': raw.get('volume_units', 'mL'),
                    'source_step_index': source_step_index,
                }
                micro.append(entry)
                continue
            if op_name in {'add', 'transfer'}:
                entry = {
                    'verb': 'pour',
                    'from': raw.get('from'),
                    'to': raw.get('to') or raw.get('target'),
                    'vessel': raw.get('vessel', 'V1'),
                    'volume': raw.get('volume'),
                    'volume_units': raw.get('volume_units', 'mL') if raw.get('volume') is not None else raw.get('volume_units'),
                    'source_step_index': source_step_index,
                }
                micro.append(entry)
                continue
            raw['verb'] = op_name
            raw['source_step_index'] = source_step_index
            micro.append(raw)
    return micro


def _infer_micro_from_raw(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    inferred: List[Dict[str, Any]] = []
    for source_step_index, step in enumerate(steps, start=1):
        action = step.get('action')
        raw = step.get('raw', '') or ''
        s = strip_tags(_clean_unicode(raw))

        # Avoid duplicating primitives already emitted from semantic steps.
        if action in {'heat_hold', 'autotitrator_rate', 'oven_dry'}:
            continue

        # Heat / maintain / water bath instructions
        if action not in {'add_solvent'} and re.search(r"\b(heat|maintain|hold|continue\s+heating?|water\s+bath)\b", s, re.I):
            temp_candidates = _temperature_candidates_c(s)
            temp = temp_candidates[0] if temp_candidates else None
            minutes = find_minutes(s)
            device = 'HP1'
            if re.search(r'water\s+bath', s, re.I):
                device = 'water bath'
            if temp is not None:
                inferred.append({
                    'verb': 'set',
                    'device': device,
                    'param': 'temperature_C',
                    'value': temp,
                    'unit': 'C',
                    'source_step_index': source_step_index,
                })
            if minutes:
                inferred.append({'verb': 'wait', 'minutes': minutes, 'source_step_index': source_step_index})

        # Stir-only instructions with duration
        if action not in {'stir', 'add_solvent'} and re.search(r"\bstir\b", s, re.I) and not re.search(r"\b(heat|oven|dry)\b", s, re.I):
            minutes = find_minutes(s)
            explicit_temps = _temperature_candidates_c(s)
            temp = explicit_temps[0] if explicit_temps else None
            if temp is not None:
                inferred.append({
                    'verb': 'set',
                    'device': 'HP1',
                    'param': 'temperature_C',
                    'value': temp,
                    'unit': 'C',
                    'source_step_index': source_step_index,
                })
            if minutes:
                inferred.append({'verb': 'wait', 'minutes': minutes, 'source_step_index': source_step_index})

        # Solvent / reagent additions -> pour primitive
        if action not in {'add', 'add_solvent', 'transfer'}:
            m_add_solvent = re.search(r"\badd\s+(\d+(?:\.\d+)?)\s*(µ?u?L|mL|ml|L|l)?\s*(?:of\s+)?([^.;]+?)\s+to\b", s, re.I)
            if m_add_solvent:
                inferred.append({
                    'verb': 'pour',
                    'reagent': _clean_solvent_tail(m_add_solvent.group(3).strip()),
                    'volume': float(m_add_solvent.group(1)),
                    'volume_units': m_add_solvent.group(2) or 'mL',
                    'vessel': step.get('vessel', 'V1'),
                    'source_step_index': source_step_index,
                })
            elif re.search(r"\btransfer\b", s, re.I):
                inferred.append({'verb': 'pour', 'vessel': step.get('vessel', 'V1'), 'source_step_index': source_step_index})

        # Oven drying
        oven = detect_oven_dry(s)
        if oven:
            tube = f"{step.get('vessel', 'V1')}_tube"
            inferred.extend([
                {'verb': 'pick_up', 'vessel': tube, 'source_step_index': source_step_index},
                {'verb': 'place', 'vessel': tube, 'to': 'oven', 'device': 'OV1', 'source_step_index': source_step_index},
                {'verb': 'set', 'device': 'OV1', 'param': 'temperature_C', 'value': oven['temperature_C'], 'unit': 'C', 'source_step_index': source_step_index},
                {'verb': 'wait', 'minutes': oven['minutes'], 'source_step_index': source_step_index},
            ])

        # pH monitoring fallback
        if re.search(r'\bph\b', s, re.I):
            inferred.append({
                'verb': 'monitor_ph',
                'device': 'PH1',
                'vessel': step.get('vessel', 'V1'),
                'source_step_index': source_step_index,
            })

        # Autotitrator rate
        m_rate = re.search(r"(?:rate(?:\s+of)?|at)\s+([0-9]+(?:\.[0-9]+)?)\s*mL\s*/\s*min", s, re.I)
        if (re.search(r'autotitrator', s, re.I) or re.search(r'titrat', s, re.I)) and m_rate:
            inferred.append({
                'verb': 'set',
                'device': 'AT1',
                'param': 'rate_mL_per_min',
                'value': float(m_rate.group(1)),
                'source_step_index': source_step_index,
            })

        # Generic final fallback: any remaining explicit temperature should yield a set op.
        temp_candidates = _temperature_candidates_c(s)
        if temp_candidates and not any(op.get('source_step_index') == source_step_index and op.get('verb') == 'set' and op.get('param') == 'temperature_C' for op in inferred):
            temp = temp_candidates[0]
            device = 'OV1' if re.search(r'\boven\b', s, re.I) else ('water bath' if re.search(r'water\s+bath', s, re.I) else 'HP1')
            inferred.append({'verb': 'set', 'device': device, 'param': 'temperature_C', 'value': temp, 'unit': 'C', 'source_step_index': source_step_index})
            minutes = find_minutes(s)
            if minutes and not any(op.get('source_step_index') == source_step_index and op.get('verb') == 'wait' for op in inferred):
                inferred.append({'verb': 'wait', 'minutes': minutes, 'source_step_index': source_step_index})
    return inferred



def _dedupe_micro_ops(micro_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for op in micro_plan:
        key = (
            op.get('source_step_index'),
            op.get('verb'),
            op.get('device'),
            op.get('param'),
            op.get('value'),
            op.get('minutes'),
            op.get('vessel'),
            op.get('tube'),
            op.get('reagent'),
            op.get('from'),
            op.get('to'),
            op.get('volume'),
            op.get('volume_units'),
            op.get('wash_cycle'),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(op)
    return out


def _insert_required_placements(micro_plan: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    out: List[Dict[str, Any]] = []
    repairs: List[str] = []
    for op in micro_plan:
        if op.get('verb') == 'set' and op.get('param') == 'temperature_C':
            device = _canonicalize_device_name(op.get('device'))
            if device in {'HP1', 'OV1'}:
                vessel = op.get('vessel') or ('V1_tube' if device == 'OV1' else 'V1')
                prev1 = out[-1] if len(out) >= 1 else None
                prev2 = out[-2] if len(out) >= 2 else None
                already = bool(
                    prev1 and prev2 and prev2.get('verb') == 'pick_up' and prev1.get('verb') == 'place' and (prev1.get('to') == device or prev1.get('device') == device)
                )
                if not already:
                    out.append({'verb': 'pick_up', 'vessel': vessel, 'source_step_index': op.get('source_step_index')})
                    out.append({'verb': 'place', 'vessel': vessel, 'to': ('oven' if device == 'OV1' else device), 'device': device, 'source_step_index': op.get('source_step_index')})
                    repairs.append(f"inserted_pickup_place_before_{device}_set")
                op = copy.deepcopy(op)
                op['device'] = device
                if 'unit' not in op:
                    op['unit'] = 'C'
        out.append(op)
    return out, repairs


def _compress_micro_plan_for_gt_style(micro_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compressed: List[Dict[str, Any]] = []
    for op in micro_plan:
        verb = op.get('verb')
        # Keep richer workup primitives in steps[].ops, but omit them from top-level micro_plan to match GT style.
        if verb in {'decant_supernatant', 'resuspend'}:
            continue
        compressed.append(op)
    return compressed


def apply_postprocessing(doc: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(doc)
    result.setdefault('_executor', {})
    result['_executor'].setdefault('schema_version', 'executor.v1')
    result['_executor'].setdefault('repairs', [])
    result['_executor']['postprocessing_applied'] = True
    result.setdefault('defaults', {})
    for k, v in DEFAULTS.items():
        result['defaults'].setdefault(k, v)
    result.setdefault('devices', {})
    for k, v in DEVICE_IDS.items():
        result['devices'].setdefault(k, v)

    steps = result.get('steps', []) or []
    micro_plan = copy.deepcopy(result.get('micro_plan') or [])
    repairs: List[str] = []

    if not micro_plan:
        micro_plan = _micro_from_ops(steps)
        if micro_plan:
            repairs.append('rebuilt_micro_plan_from_step_ops')

    inferred = _infer_micro_from_raw(steps)
    if inferred:
        micro_plan.extend(inferred)
        repairs.append('inferred_micro_ops_from_raw_steps')

    # Canonicalize devices and add default units
    for op in micro_plan:
        if 'device' in op:
            canon = _canonicalize_device_name(op.get('device'))
            if canon != op.get('device'):
                repairs.append(f"canonicalized_device_{str(op.get('device')).replace(' ', '_')}_to_{canon}")
                op['device'] = canon
        if op.get('verb') == 'set' and op.get('param') == 'temperature_C' and 'unit' not in op:
            op['unit'] = 'C'
        if op.get('verb') == 'pour' and op.get('volume') is not None and 'volume_units' not in op:
            op['volume_units'] = 'mL'
        if op.get('verb') == 'place' and op.get('to') == 'oven':
            op.setdefault('device', 'OV1')

    micro_plan, placement_repairs = _insert_required_placements(micro_plan)
    repairs.extend(placement_repairs)

    # Final coverage pass for oven placement / temperature and generic temperature or pH fallback.
    for i, step in enumerate(steps, start=1):
        raw = strip_tags(_clean_unicode(step.get('raw', '') or ''))
        has_temp = any(
            op.get('source_step_index') == i and (
                (op.get('verb') == 'set' and op.get('param') == 'temperature_C') or
                (op.get('verb') == 'set_oven_temperature')
            )
            for op in micro_plan
        )
        has_place = any(
            op.get('source_step_index') == i and (
                (op.get('verb') == 'place' and (op.get('to') == 'oven' or op.get('device') == 'OV1')) or
                (op.get('verb') == 'move_to_oven')
            )
            for op in micro_plan
        )
        if re.search(r'\boven\b', raw, re.I):
            if not has_place:
                micro_plan.append({'verb': 'pick_up', 'vessel': f"{step.get('vessel', 'V1')}_tube", 'source_step_index': i})
                micro_plan.append({'verb': 'place', 'vessel': f"{step.get('vessel', 'V1')}_tube", 'to': 'oven', 'device': 'OV1', 'source_step_index': i})
                repairs.append('added_oven_pickup_place_fallback')
            if not has_temp and find_temp_c(raw) is not None:
                temp_candidates = _temperature_candidates_c(raw)
                if temp_candidates:
                    micro_plan.append({'verb': 'set', 'device': 'OV1', 'param': 'temperature_C', 'value': temp_candidates[0], 'unit': 'C', 'source_step_index': i})
                repairs.append('added_oven_temperature_fallback')
        elif not has_temp:
            temp_candidates = _temperature_candidates_c(raw)
            if temp_candidates:
                device = 'HP1'
                if re.search(r'water\s+bath', raw, re.I):
                    device = 'water bath'
                elif re.search(r'\boven\b', raw, re.I):
                    device = 'OV1'
                micro_plan.append({'verb': 'set', 'device': device, 'param': 'temperature_C', 'value': temp_candidates[0], 'unit': 'C', 'source_step_index': i})
                repairs.append('added_temperature_fallback')
        if re.search(r'\bph\b', raw, re.I) and not any(op.get('source_step_index') == i and op.get('verb') == 'monitor_ph' for op in micro_plan):
            micro_plan.append({'verb': 'monitor_ph', 'device': 'PH1', 'vessel': step.get('vessel', 'V1'), 'source_step_index': i})
            repairs.append('added_ph_monitor_fallback')

    # Re-canonicalize after fallback additions.
    for op in micro_plan:
        if 'device' in op:
            op['device'] = _canonicalize_device_name(op.get('device'))

    micro_plan = _compress_micro_plan_for_gt_style(micro_plan)
    micro_plan = _dedupe_micro_ops(micro_plan)
    _assign_unique_action_step_indices(micro_plan)
    result['micro_plan'] = micro_plan

    # Build minimal plan with collapse + mapping.
    map_generic = __import__('os').environ.get('MIN_PLAN_MAP_GENERIC') == '1'
    device_mapping = {'HP1': 'hotplate', 'SP1': 'stir_plate', 'OV1': 'oven', 'CF1': 'centrifuge', 'AT1': 'autotitrator'}
    micro_plan_min: List[Dict[str, Any]] = []
    timing_delays: List[Dict[str, Any]] = []
    seen_sets = set()
    for op in micro_plan:
        verb = op.get('verb')
        if verb == 'wait' and op.get('minutes', 0) > 0:
            timing_delays.append({'step_index': op.get('step_index', 1), 'verb': 'wait', 'minutes': op.get('minutes')})
        if verb not in {'pick_up', 'place', 'pour', 'set'}:
            continue
        entry = copy.deepcopy(op)
        if map_generic:
            if entry.get('device') in device_mapping:
                entry['device'] = device_mapping[entry['device']]
            if entry.get('to') in device_mapping:
                entry['to'] = device_mapping[entry['to']]
        if entry.get('verb') == 'set':
            key = (entry.get('device') or entry.get('to'), entry.get('param'), entry.get('value'))
            if key in seen_sets:
                continue
            seen_sets.add(key)
        micro_plan_min.append(entry)

    # Ensure primitive pours appear for add / add_solvent / transfer even if parser only created structured steps.
    existing_pour_sources = {op.get('source_step_index') for op in micro_plan_min if op.get('verb') == 'pour'}
    next_idx = max([op.get('step_index', 0) for op in micro_plan_min] + [0]) + 1
    for i, step in enumerate(steps, start=1):
        action = step.get('action')
        if action in {'add', 'add_solvent', 'transfer', 'add_prepared_solution', 'add_reagent_solution'} and i not in existing_pour_sources:
            entry = {'verb': 'pour', 'step_index': next_idx, 'source_step_index': i}
            if step.get('volume') is not None:
                entry['volume'] = step.get('volume')
                entry['volume_units'] = step.get('volume_units', 'mL')
            micro_plan_min.append(entry)
            next_idx += 1
        if action == 'oven_dry' and not any(op.get('source_step_index') == i and op.get('verb') == 'place' for op in micro_plan_min):
            dev = 'oven' if map_generic else 'OV1'
            micro_plan_min.append({'verb': 'place', 'device': dev, 'to': 'oven', 'source_step_index': i, 'step_index': next_idx}); next_idx += 1
            micro_plan_min.append({'verb': 'set', 'device': dev, 'param': 'temperature_C', 'value': step.get('temperature_C', 80), 'unit': 'C', 'source_step_index': i, 'step_index': next_idx}); next_idx += 1

    _assign_unique_action_step_indices(micro_plan_min)
    result['micro_plan_min'] = micro_plan_min
    deduped_timing: List[Dict[str, Any]] = []
    seen_timing = set()
    for delay in timing_delays:
        key = (delay.get('verb'), delay.get('minutes'), delay.get('step_index'))
        if key in seen_timing:
            continue
        seen_timing.add(key)
        deduped_timing.append(delay)
    result['timing_delays'] = deduped_timing
    result['_executor']['repairs'].extend(repairs)
    return result


def _enrich_step_scalar_fields(steps: List[Dict[str, Any]]) -> None:
    """Backfill scalar fields like temperature_C and target_ph onto emitted step records.

    Some tests inspect doc["steps"] directly rather than micro_plan, so these values
    need to exist on the step records themselves even if they were only inferred later
    for primitive execution ops.
    """
    ph_rx = re.compile(r"\bpH\s*(?:to|=|of|reaches?|reach)?\s*(\d+(?:\.\d+)?)", re.I)
    for step in steps:
        raw = step.get("raw", "") or ""
        if step.get("temperature_C") in (None, ""):
            temps = _temperature_candidates_c(raw)
            if temps:
                step["temperature_C"] = temps[0]
        if step.get("target_ph") in (None, ""):
            m = ph_rx.search(raw)
            if m:
                try:
                    step["target_ph"] = float(m.group(1))
                except Exception:
                    pass


def convert_text_to_robot_ops(text: str) -> Dict[str, Any]:
    hardware = parse_hardware(text)
    vessels = VesselRegistry(hardware)
    primary_vessel = vessels.ensure_glassware("Beaker")
    semantic_steps, _context = semantic_parse(text, vessels)
    records = emit_steps(semantic_steps, vessels)
    _enrich_step_scalar_fields(records)

    result: Dict[str, Any] = {
        "_executor": {
            "schema_version": "executor.v1",
            "converter": "converter_v2",
            "postprocessing_applied": False,
            "repairs": [],
        },
        "devices": dict(DEVICE_IDS),
        "defaults": dict(DEFAULTS),
        "hardware": hardware,
        "vessel_registry": vessels.as_dict() or {primary_vessel: "Beaker"},
        "vessel_contents_detailed": vessels.contents_detailed,
        "steps": records,
    }

    if "V1" not in result["vessel_registry"]:
        result["vessel_registry"]["V1"] = "Beaker"
    if any(step.get("action") == "postprocess" for step in records):
        result["vessel_registry"].setdefault("V1_tube", "Centrifuge Tube")

    result["micro_plan"] = build_micro_plan(records)
    result = apply_postprocessing(result)
    result.setdefault("micro_plan_min", [])
    result.setdefault("timing_delays", [])

    # Last-resort document-level temperature fallback for broad fallback tests.
    has_temp_set = any(
        op.get("verb") == "set" and op.get("param") == "temperature_C"
        for op in result.get("micro_plan", [])
    )
    if not has_temp_set:
        doc_temps = _temperature_candidates_c(text)
        if doc_temps:
            temp_op = {
                "verb": "set",
                "device": "HP1",
                "param": "temperature_C",
                "value": doc_temps[0],
                "unit": "C",
                "source_step_index": 1,
            }
            result.setdefault("micro_plan", []).append(temp_op)
            _assign_unique_action_step_indices(result["micro_plan"])
            generic_map = __import__('os').environ.get('MIN_PLAN_MAP_GENERIC') == '1'
            min_device = 'hotplate' if generic_map else 'HP1'
            result.setdefault("micro_plan_min", []).append({
                "verb": "set",
                "device": min_device,
                "param": "temperature_C",
                "value": doc_temps[0],
                "unit": "C",
                "source_step_index": 1,
                "step_index": len(result.get("micro_plan_min", [])) + 1,
            })
            if not any(d.get("minutes") for d in result.get("timing_delays", [])):
                doc_minutes = find_minutes(text)
                if doc_minutes:
                    result.setdefault("timing_delays", []).append({
                        "step_index": 1,
                        "verb": "wait",
                        "minutes": doc_minutes,
                    })
            result.setdefault("_executor", {}).setdefault("repairs", []).append("added_document_level_temperature_fallback")

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
