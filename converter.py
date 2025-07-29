from __future__ import annotations

import json, re, math, textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Tuple

__all__ = ["convert_to_json", "ParserError"]
SCHEMA_VERSION = "1.8"

class ParserError(ValueError):
    pass

# ---------- tolerant headings ----------
_HEADING_LINE = re.compile(
    r"""^\s*
        (?:\d+[\.\)]\s*)?
        (?:\*\*)?
        (?P<name>
            hardware(?:\s*&\s*glassware)? |
            glassware |
            materials |
            reagents |
            procedure |
            steps |
            method
        )
        (?:\*\*)?
        \s*:?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_LIST_BULLET = re.compile(r"^\s*(?:[-*•–—]\s+|\d+[\.\)]\s+)")

# quantities / units
_AMOUNT_RX = re.compile(r"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg|µl|μl|ul|mL|ml|L|l)\b", re.I)
_CONC_RX   = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mM|M|%(?:\s*w\/v|\s*v\/v)?)\b", re.I)
_VOL_RX    = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<vunit>mL|L)\b", re.I)

# --- robot-mode cleanup + meta detection ---
_REASONING_MARKERS = re.compile(
    r"\b(because|since|so that|therefore|thus|hence|rationale|justif(?:y|ication)|note:|ensuring)\b",
    re.I,
)
_META_RX = re.compile(
    r"^\s*(this (protocol|procedure)|these steps|the following (procedure|protocol)|"
    r"which can be|intended to|overview|background|this will serve|this approach|"
    r"this step is crucial|ensure stability)\b",
    re.I,
)
_STRIP_CIT_RX = re.compile(r"\s*\[(?:CTX|GEN|\d+)\]\s*$", re.I)
_TRAIL_REASON_RX = re.compile(
    r"\s*(?:[,;.\-–—]\s*)?(?:to (?:ensure|facilitate|promote|allow)\b.*|ensuring\b.*)$",
    re.I,
)

# --- vessel detection/labeling ---
_VESSEL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"schlenk flask", re.I), "schlenk_flask"),
    (re.compile(r"round[- ]?bottom flask|rbf", re.I), "flask"),
    (re.compile(r"\bflask\b", re.I), "flask"),
    (re.compile(r"\bbeaker\b", re.I), "beaker"),
    (re.compile(r"\bvial\b", re.I), "vial"),
    (re.compile(r"centrifuge tube|eppendorf|falcon", re.I), "tube"),
    (re.compile(r"\btube\b", re.I), "tube"),
    (re.compile(r"\breaction vessel|reactor|autoclave|teflon-lined", re.I), "reactor"),
]
_SIZE_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(mL|L)\b", re.I)
_SEPARATE_VESSEL_RX = re.compile(
    r"\b(?:in(?:to)?\s+(?:a|an)\s+)"
    r"(?:(?P<vol>\d+(?:\.\d+)?)\s*(?P<vu>mL|L)\s+)?"
    r"(?P<typ>schlenk flask|round[- ]?bottom flask|flask|beaker|vial|centrifuge tube|tube)\b",
    re.I,
)

# --- “prepare X g Y in Z mL solvent” → atomic steps ---
_PREPARE_SOLUTION_RX = re.compile(
    r"\b(prepare|make|formulate)\s+"
    r"(?P<amt>\d+(?:\.\d+)?)\s*(?P<aunit>mg|g|kg)\s+"
    r"(?P<chem>[^,;.\n]+?)\s+in\s+"
    r"(?P<vol>\d+(?:\.\d+)?)\s*(?P<vunit>mL|L)\s+"
    r"(?P<solv>[^.;\n]+)",
    re.I,
)

# --- pH control expansion ---
_PH_RX = re.compile(r"pH\s*(?:of\s*)?(?:around\s*)?(?P<val>\d+(?:\.\d+)?)", re.I)

# --- stir speed extraction (prefer procedure context; else default 900 rpm) ---
_STIR_SPEED_RXS = [
    re.compile(r"\bstir(?:ring)?\s*(?:at|speed\s*(?:to|=)?)\s*(?P<rpm>\d{2,4})\s*rpm\b", re.I),
    re.compile(r"\bset\s*stirr(?:er|ing)\s*(?:speed\s*(?:to|=)?)?\s*(?P<rpm>\d{2,4})\s*rpm\b", re.I),
]

# --- ultrasonic and time normalization helpers ---
def _norm_time(val: str, unit: Optional[str]) -> str:
    v = val.strip()
    u = (unit or "").lower()
    if u in ("h","hr","hrs","hour","hours"): return f"{v} h"
    if u in ("s","sec","secs","second","seconds"): return f"{v} s"
    return f"{v} min"  # default minutes

# Setup lines to dedupe globally
_SETUP_DEDUP_RXS = [
    re.compile(r"^insert stir bar into vessel$", re.I),
    re.compile(r"^place vessel on magnetic stir plate$", re.I),
    re.compile(r"^turn on stir plate to target speed$", re.I),
]

# --- NEW: detect RCF and radius (for RPM conversion) ---
_RCF_RX = re.compile(r"(?:\brcf\b|\b(\d{3,6})\s*(?:×\s*)?g|\b\d{3,6}\s*xg)\b", re.I)
# Simpler targeted extractor to pull numeric value near 'rcf' or 'g'
_RCF_VALUE_RX = re.compile(r"(?P<rcf>\d{3,6})\s*(?:×\s*)?(?:g|xg)\b|\brcf\b\s*(?P<rcf2>\d{3,6})", re.I)
_RADIUS_RX = re.compile(r"(?:radius|r\s*=?)\s*(?P<rad>\d+(?:\.\d+)?)\s*cm", re.I)

def _rcf_to_rpm(rcf: float, r_cm: float) -> int:
    # rpm = sqrt( RCF / (1.118e-5 * r_cm) )
    rpm = math.sqrt(rcf / (1.118e-5 * r_cm))
    return int(round(rpm))

def _extract_rcf_and_radius(text: str) -> tuple[Optional[float], Optional[float]]:
    rcf = None
    rad = None
    m = _RCF_VALUE_RX.search(text)
    if m:
        rcf_str = m.group("rcf") or m.group("rcf2")
        try:
            rcf = float(rcf_str)
        except Exception:
            rcf = None
    rm = _RADIUS_RX.search(text)
    if rm:
        try:
            rad = float(rm.group("rad"))
        except Exception:
            rad = None
    return rcf, rad

# Global fallbacks to pull rpm/time from the whole text (for centrifuge)
_CENT_RPM_GLOBAL = re.compile(r"centrifug\w*[^.]*?\bat\s*(?P<rpm>\d{3,5})\s*rpm", re.I)
_CENT_TIME_GLOBAL = re.compile(
    r"centrifug\w*[^.]*?\bfor\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<u>min|mins|minutes|minute|h|hr|hrs|hour|hours|s|sec|secs|seconds)\b",
    re.I
)

def _extract_centrifuge_time(text: str) -> Optional[str]:
    m = _CENT_TIME_GLOBAL.search(text)
    if m:
        return _norm_time(m.group("val"), m.group("u"))
    return None

def _extract_centrifuge_params(text: str) -> tuple[Optional[int], Optional[str]]:
    """Find rpm OR RCF+radius anywhere, plus a time."""
    # 1) explicit rpm
    m = _CENT_RPM_GLOBAL.search(text)
    if m:
        try:
            rpm = int(m.group("rpm"))
            return rpm, _extract_centrifuge_time(text)
        except ValueError:
            pass
    # 2) RCF (+ optional radius)
    rcf, rad = _extract_rcf_and_radius(text)
    if rcf is not None:
        r_cm = rad if rad is not None else 11.0
        rpm = _rcf_to_rpm(rcf, r_cm)
        return rpm, _extract_centrifuge_time(text)
    # 3) at least return time if found
    return None, _extract_centrifuge_time(text)

def _protocol_window(text: str) -> str:
    m = re.search(r"^\s*##\s*SynthesisProtocol\b.*", text, flags=re.I | re.M | re.S)
    return text[m.start():] if m else text

def _canon(name: str) -> str:
    n = name.lower()
    if "hardware" in n or "glassware" in n: return "hardware"
    if n in ("materials", "reagents"):      return "reagents"
    return "procedure"

def _split_sections_tolerant(text: str) -> Dict[str, List[str]]:
    text = _protocol_window(text)
    buckets: Dict[str, List[str]] = {"hardware": [], "reagents": [], "procedure": []}
    current: Optional[str] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip(): continue
        norm = re.sub(r"[\*_`]", "", line).strip()
        m = _HEADING_LINE.match(norm)
        if m:
            current = _canon(m.group("name")); continue
        if current: buckets[current].append(raw)
        else:       buckets.setdefault("_pre", []).append(raw)
    return buckets

def _bullets(lines: List[str]) -> List[str]:
    if not lines: return []
    any_bullets = any(_LIST_BULLET.match(l) for l in lines)
    items: List[str] = []
    for l in lines:
        s = l.strip()
        if not s: continue
        if any_bullets: s = _LIST_BULLET.sub("", s).strip()
        items.append(s)
    return items

# ---------- reagents ----------
@dataclass
class Reagent:
    description: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    concentration: Optional[str] = None
    final_volume_mL: Optional[float] = None
    def asdict(self): return asdict(self)

def _parse_reagent(line: str) -> Reagent:
    amt  = _AMOUNT_RX.search(line)
    conc = _CONC_RX.search(line)
    vol  = _VOL_RX.search(line)
    return Reagent(
        description=line,
        amount=float(amt.group("qty")) if amt else None,
        unit=(amt.group("unit") if amt else None),
        concentration=(conc.group(0) if conc else None),
        final_volume_mL=(float(vol.group("val")) if vol else None),
    )

# ---------- expansion rules ----------
Action = List[str]
RuleFn = Callable[[re.Match], Action]

def _centrifuge_actions_from_match(m: re.Match) -> List[str]:
    # Prefer explicit rpm in the same line
    rpm_val: Optional[int] = None
    if m.groupdict().get("rpm"):
        try:
            rpm_val = int(m.group("rpm"))
        except Exception:
            rpm_val = None

    # If no rpm, try RCF + radius in the matched text
    if rpm_val is None:
        rcf, rad = _extract_rcf_and_radius(m.group(0))
        if rcf is not None:
            r_cm = rad if rad is not None else 11.0
            rpm_val = _rcf_to_rpm(rcf, r_cm)

    # Time (if present in this line)
    tstr: Optional[str] = None
    if m.groupdict().get("tval"):
        tstr = _norm_time(m.group("tval"), m.group("tunit"))

    rpm_txt = f"set speed {rpm_val} rpm" if rpm_val else "set speed as specified"
    time_txt = f"run for {tstr}" if tstr else "run for specified time"
    return [
        "load tubes into centrifuge",
        rpm_txt,
        time_txt,
        "pour off supernatant into waste, keep pellet",
    ]

ACTION_RULES: List[Tuple[re.Pattern, RuleFn]] = [
    # Ultrasonic bath (put BEFORE transfer so it wins)
    (re.compile(r"\bultrasonic bath\b.*?\bfor\s*(?P<time>\d+(?:\.\d+)?)\s*(?P<tunit>min|mins|minutes|s|sec|secs|seconds)\b", re.I),
     lambda m: ["place container in ultrasonic bath",
                f"run sonication for {_norm_time(m.group('time'), m.group('tunit'))}"]),

    # Stir
    (re.compile(r"\bstir(?:red|ring)?\b", re.I),
     lambda m: ["insert stir bar into vessel",
                "place vessel on magnetic stir plate",
                "turn on stir plate to target speed"]),

    # Add X
    (re.compile(r"\badd(?:ed)?\s+(.*)", re.I),
     lambda m: [f"add {m.group(1).strip()} to vessel"]),

    # Transfer … to …
    (re.compile(r"\btransfer(?:red)?\s+(.*)\s+to\s+(.*)", re.I),
     lambda m: [f"transfer {m.group(1).strip()} to {m.group(2).strip()}"]),

    # Heat to T for time  (captures both T and time)
    (re.compile(r"\bheat(?:ed)?\s+(?:the\s+)?(?:mixture|solution|suspension)\s*to\s*(?P<T>\d+(?:\.\d+)?)\s*°?\s*C(?:[^\.]*?)\bfor\s*(?P<tval>\d+(?:\.\d+)?)\s*(?P<tunit>min|mins|minutes|h|hr|hrs|hour|hours)\b", re.I),
     lambda m: [f"set heating device to {m.group('T')} °C",
                "monitor temperature until set point reached",
                f"hold temperature at {m.group('T')} °C for {_norm_time(m.group('tval'), m.group('tunit'))}"]),

    # Heat to T (no time)
    (re.compile(r"\bheat(?:ed)?\s+to\s+(?P<T>\d+(?:\.\d+)?)\s*°?\s*C\b", re.I),
     lambda m: [f"set heating device to {m.group('T')} °C",
                "monitor temperature until set point reached"]),

    # Maintain at T for time
    (re.compile(r"\bmaintain(?:ed)?\s+at\s+(?P<T>\d+(?:\.\d+)?)\s*°?\s*C\s+for\s+(?P<tval>\d+(?:\.\d+)?)\s*(?P<tunit>min|mins|minutes|h|hr|hrs|hour|hours)\b", re.I),
     lambda m: [f"hold temperature at {m.group('T')} °C for {_norm_time(m.group('tval'), m.group('tunit'))}"]),

    # Cool to T °C
    (re.compile(r"\bcool(?:ed)?\s+to\s+(?P<T>\d+(?:\.\d+)?)\s*°?\s*C\b", re.I),
     lambda m: [f"cool vessel to {m.group('T')} °C"]),

    # Centrifuge — supports explicit rpm or RCF (×g/rcf) and time
    (re.compile(
        r"\bcentrifug(e|ed|ation)\b"
        r".*?(?:(?:at|@)\s*(?P<rpm>\d{2,5})\s*rpm)?"
        r"(?:.*?(?:rcf\b|\d{3,6}\s*(?:×\s*)?g|\d{3,6}\s*xg).*?)?"
        r"(?:.*?\bfor\s*(?P<tval>\d+(?:\.\d+)?)(?:\s*(?P<tunit>min|mins|minutes|minute|h|hr|hrs|hour|hours|s|sec|secs|seconds))?)?",
        re.I),
     _centrifuge_actions_from_match),

    # Purge with gas
    (re.compile(r"\b(purge|degass?|bubble)\s+(with\s+)?(n2|nitrogen|argon|ar)\b", re.I),
     lambda m: ["connect inert gas line to vessel",
                "open gas flow to purge headspace",
                "maintain flow for specified duration"]),

    # Vacuum dry
    (re.compile(r"\b(dry|evaporate)\b.*\b(vacuum|vac)\b", re.I),
     lambda m: ["place sample in vacuum chamber",
                "apply vacuum until solvent removed or mass constant"]),

    # pH adjust (generic)
    (re.compile(r"\badjust\s+pH\s+to\s+(\d+(?:\.\d+)?)", re.I),
     lambda m: [f"measure solution pH",
                f"add acid/base to reach pH {m.group(1)}",
                "verify pH is stable"]),

    # Sonication (generic, if not matched by timed rule)
    (re.compile(r"\bsonicat(e|ed|ion)\b", re.I),
     lambda m: ["place container in ultrasonic bath",
                "run sonication for specified time / power"]),

    # Wash
    (re.compile(r"\bwash(?:ed)?\s+with\s+(.*)", re.I),
     lambda m: [f"wash solid with {m.group(1).strip()}",
                "discard washings or combine as specified"]),
]

def _expand_procedure_line(line: str) -> List[str]:
    for rx, fn in ACTION_RULES:
        m = rx.search(line)
        if m:
            return fn(m)
    return [line]

# ---------- robot-mode helpers ----------
def _strip_reasoning_fragments(s: str) -> str:
    s = _STRIP_CIT_RX.sub("", s)
    s = re.sub(r"^\s*finally,\s*", "", s, flags=re.I)
    s = re.sub(r"\((?:[^)]{0,80})\)", lambda m: ("" if _REASONING_MARKERS.search(m.group(0)) else m.group(0)), s)
    s = _TRAIL_REASON_RX.sub("", s)
    s = re.sub(r"[:–—-]\s*(?=(because|since|so that|therefore|thus|hence)\b).*", "", s, flags=re.I)
    s = re.sub(r"\b(because|since|so that|therefore|thus|hence|rationale|justif(?:y|ication)|note:|ensuring).*$", "", s, flags=re.I)
    s = re.sub(r"\.\s*to vessel\s*$", "", s, flags=re.I)
    return s.strip()

def _is_meta_line(s: str) -> bool:
    return bool(_META_RX.search(s))

def _normalize_prepare_line(s: str) -> str:
    return re.sub(r"^\s*in a separate (container|vessel),\s*", "", s, flags=re.I).strip()

def _normalize_dry_line(s: str) -> Optional[str]:
    m = re.search(r"\bdry\b.*?\bat\s*(\d+(?:\.\d+)?)\s*°?\s*C.*?\bfor\s*([^.;,\n]+)", s, flags=re.I)
    if not m:
        return None
    T = m.group(1); t = re.sub(r"\s+", " ", m.group(2)).strip()
    t = (t.replace("hours", "h").replace("hour", "h")
            .replace("mins", "min").replace("minutes", "min"))
    return f"dry sample in oven at {T} °C for {t}"

def _cleanup_add_target(s: str) -> str:
    if s.lower().startswith("add ") and " to vessel" in s.lower() and re.search(r"\bto\s+(?!vessel)\b", s, re.I):
        s = re.sub(r"\s+to vessel\s*$", "", s, flags=re.I)
    return s

def _expand_prepare_solution(line: str) -> Optional[List[str]]:
    m = _PREPARE_SOLUTION_RX.search(line)
    if not m:
        return None
    amt, aunit = m.group("amt"), m.group("aunit")
    chem = re.sub(r"\s+", " ", (m.group("chem") or "").strip())
    vol, vunit = m.group("vol"), m.group("vunit")
    solv = re.sub(r"\s+", " ", (m.group("solv") or "").strip())
    return [
        "place clean vessel on balance",
        "tare balance",
        f"weigh {amt} {aunit} {chem}",
        f"add {vol} {vunit} {solv} to vessel",
        f"add {amt} {aunit} {chem} to vessel",
        "insert stir bar into vessel",
        "place vessel on magnetic stir plate",
        "turn on stir plate to target speed",
        "stir until dissolved",
    ]

def _expand_pH_adjust(line: str) -> Optional[List[str]]:
    if re.search(r"\bnaoh\b", line, re.I) and (m := _PH_RX.search(line)):
        pH = m.group("val")
        core = re.sub(r"\s*while\s+maintain.*$", "", line, flags=re.I)
        core = re.sub(r"\s*,?\s*which\s+promotes.*$", "", core, flags=re.I)
        core = re.sub(r"\s*\[(?:CTX|GEN|\d+)\]\s*$", "", core)
        return [
            _cleanup_add_target(_strip_reasoning_fragments(core)).rstrip("."),
            f"monitor pH (target {pH} ± 0.2)",
            f"adjust with NaOH to reach pH {pH}",
        ]
    return None

def _detect_vessel_type(s: str) -> Optional[str]:
    for rx, vtype in _VESSEL_PATTERNS:
        if rx.search(s): return vtype
    return None

def _ensure_vessel(vessels: List[Dict[str, str]], desc: str) -> str:
    for v in vessels:
        if v["description"].lower() == desc.lower():
            return v["id"]
    vid = f"V{len(vessels)+1}"
    vtype = (desc.split()[-1] if desc.split() else "vessel")
    vessels.append({"id": vid, "type": vtype, "description": desc})
    return vid

def _collect_vessels_from_hardware(hardware: List[str]) -> List[Dict[str, str]]:
    vessels: List[Dict[str, str]] = []
    for h in hardware:
        vtype = _detect_vessel_type(h or "")
        if vtype:
            size = _SIZE_RX.search(h or "")
            desc = (f"{size.group(0)} " if size else "") + vtype.replace("_", " ")
            _ensure_vessel(vessels, desc.strip())
    if not vessels:
        _ensure_vessel(vessels, "flask")  # default V1
    return vessels

def _dedupe_adjacent(seq: List[str]) -> List[str]:
    out: List[str] = []
    last = None
    for s in seq:
        if s != last:
            out.append(s)
            last = s
    return out

def _global_dedupe_setup(seq: List[str]) -> List[str]:
    seen = [False] * len(_SETUP_DEDUP_RXS)
    out: List[str] = []
    for s in seq:
        lowered = s.strip().lower()
        matched_setup = False
        for i, rx in enumerate(_SETUP_DEDUP_RXS):
            if rx.match(lowered):
                matched_setup = True
                if not seen[i]:
                    out.append(s); seen[i] = True
                break
        if not matched_setup:
            out.append(s)
    return out

def _extract_stir_speed(text: str) -> int:
    for rx in _STIR_SPEED_RXS:
        m = rx.search(text)
        if m:
            try:
                rpm = int(m.group("rpm"))
                if 20 <= rpm <= 5000:
                    return rpm
            except ValueError:
                pass
    return 900  # default if not specified

def _assign_targets(steps: List[str], vessels: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not vessels:
        vessels = [{"id": "V1", "type": "flask", "description": "flask"}]
    current = vessels[0]["id"]
    out: List[Dict[str, str]] = []
    for s in steps:
        m = _SEPARATE_VESSEL_RX.search(s)
        if m:
            vol = (m.group("vol") or "").strip()
            vu  = (m.group("vu")  or "").strip()
            typ = (m.group("typ") or "").strip().lower()
            desc = (f"{vol} {vu} " if vol and vu else "") + typ
            desc = re.sub(r"\s+", " ", desc).strip()
            current = _ensure_vessel(vessels, desc)
        if re.search(r"\bcentrifug", s, re.I):
            # Ensure a tube exists; then use it
            tube = next((v for v in vessels if "tube" in v["description"].lower()), None)
            if not tube:
                vid = _ensure_vessel(vessels, "tube")
                tube = next(v for v in vessels if v["id"] == vid)
            current = tube["id"]
        out.append({"action": s, "target": current})
    return out

# ---------- main ----------
def convert_to_json(raw: str, robot: bool = False) -> Dict[str, object]:
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")

    raw = textwrap.dedent(raw).strip()
    sections = _split_sections_tolerant(raw)
    hardware_lines  = _bullets(sections.get("hardware", []))
    reagent_lines   = _bullets(sections.get("reagents", []))
    procedure_lines = _bullets(sections.get("procedure", []))
    if not any([hardware_lines, reagent_lines, procedure_lines]):
        procedure_lines = _bullets(sections.get("_pre", [])) or _bullets(raw.splitlines())

    reagents = [_parse_reagent(l).asdict() for l in reagent_lines]

    # Expand to low-level actions first
    expanded: List[str] = []
    for line in procedure_lines:
        # add specific expanders before generic rules
        if (prep := _expand_prepare_solution(line)):
            expanded.extend(prep); continue
        if (ph := _expand_pH_adjust(line)):
            expanded.extend(ph); continue
        expanded.extend(_expand_procedure_line(line))

    vessels = _collect_vessels_from_hardware(hardware_lines)

    final_actions = expanded
    if robot:
        tmp: List[str] = []
        for s in expanded:
            if _is_meta_line(s):
                continue
            s = _normalize_prepare_line(s)

            # Specific normalizers
            if (dry := _normalize_dry_line(s)):
                tmp.append(dry); continue

            s2 = _cleanup_add_target(_strip_reasoning_fragments(s))
            if s2:
                tmp.append(s2.rstrip("."))

        tmp = _dedupe_adjacent(tmp)
        tmp = _global_dedupe_setup(tmp)
        final_actions = tmp

        # Replace "target speed" with discovered or default rpm
        chosen_rpm = _extract_stir_speed(raw) or _extract_stir_speed(" ".join(procedure_lines))
        final_actions = [
            (re.sub(r"\bturn on stir plate to target speed\b",
                    f"set stirrer to {chosen_rpm} rpm", a, flags=re.I))
            for a in final_actions
        ]

        # Fill centrifuge placeholders from global values, if present
        rpm, ctime = _extract_centrifuge_params(raw)
        if rpm or ctime:
            new = []
            for a in final_actions:
                a2 = a
                if rpm:
                    a2 = re.sub(r"\bset speed as specified\b", f"set speed {rpm} rpm", a2, flags=re.I)
                if ctime:
                    a2 = re.sub(r"\brun for specified time\b", f"run for {ctime}", a2, flags=re.I)
                new.append(a2)
            final_actions = new

    structured = _assign_targets(final_actions if robot else expanded, vessels)
    vessel_map = {v["id"]: v["description"] for v in vessels}

    out: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "title"         : raw.split("\n", 1)[0].lstrip("# ")[:120],
        "hardware"      : hardware_lines,
        "reagents"      : reagents,
        "procedure"     : final_actions if robot else expanded,
        "characterization": [],
        "storage"       : "",
    }
    if robot:
        out["vessels"] = vessels
        out["vessel_map"] = vessel_map
        out["procedure_structured"] = structured
        out["robot"] = {"cleaned": True, "vessel_labels": True, "atomic_prepare": True, "rpm_default": 900}

    if not (out["reagents"] or out["procedure"] or out.get("hardware")):
        raise ParserError("Could not recognize any sections or steps.")
    return out

if __name__ == "__main__":
    import sys
    print(json.dumps(convert_to_json(open(sys.argv[1], "r", encoding="utf-8").read(), robot=True),
                     indent=2, ensure_ascii=False))
