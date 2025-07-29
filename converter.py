# converter.py
from __future__ import annotations

import json, re, textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Tuple

__all__ = ["convert_to_json", "ParserError"]
SCHEMA_VERSION = "1.5"

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
# Drop whole lines that are meta commentary rather than actions
_META_RX = re.compile(
    r"^\s*(this (protocol|procedure)|these steps|the following (procedure|protocol)|"
    r"which can be|intended to|overview|background|this will serve|this approach)\b",
    re.I,
)
# Strip citation-like tail tokens
_STRIP_CIT_RX = re.compile(r"\s*\[(?:CTX|GEN|\d+)\]\s*$", re.I)
# Strip trailing rationale clauses (“to ensure … / to facilitate … / ensuring …”)
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
# Detect “in a/into a <size?> <vessel>”
_SEPARATE_VESSEL_RX = re.compile(
    r"\b(?:in(?:to)?\s+(?:a|an)\s+)"
    r"(?:(?P<vol>\d+(?:\.\d+)?)\s*(?P<vu>mL|L)\s+)?"
    r"(?P<typ>schlenk flask|round[- ]?bottom flask|flask|beaker|vial|centrifuge tube|tube)\b",
    re.I,
)

# Detect “prepare X g Y in Z mL SOLVENT”
_PREPARE_SOLUTION_RX = re.compile(
    r"\b(prepare|make|formulate)\s+"
    r"(?P<amt>\d+(?:\.\d+)?)\s*(?P<aunit>mg|g|kg)\s+"
    r"(?P<chem>[^,;.\n]+?)\s+in\s+"
    r"(?P<vol>\d+(?:\.\d+)?)\s*(?P<vunit>mL|L)\s+"
    r"(?P<solv>[^.;\n]+)",
    re.I,
)

# Setup lines we’ll dedupe globally (keep first occurrence; still keep repeated WASH steps)
_SETUP_DEDUP_RXS = [
    re.compile(r"^insert stir bar into vessel$", re.I),
    re.compile(r"^place vessel on magnetic stir plate$", re.I),
    re.compile(r"^turn on stir plate to target speed$", re.I),
    re.compile(r"^connect inert gas line to vessel$", re.I),
    re.compile(r"^open gas flow to purge headspace$", re.I),
]

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
ACTION_RULES: List[Tuple[re.Pattern, RuleFn]] = [
    (re.compile(r"\bstir(?:red|ring)?\b", re.I),
     lambda m: ["insert stir bar into vessel",
                "place vessel on magnetic stir plate",
                "turn on stir plate to target speed"]),
    (re.compile(r"\badd(?:ed)?\s+(.*)", re.I),
     lambda m: [f"add {m.group(1).strip()} to vessel"]),
    (re.compile(r"\btransfer(?:red)?\s+(.*)\s+to\s+(.*)", re.I),
     lambda m: [f"transfer {m.group(1).strip()} to {m.group(2).strip()}"]),
    (re.compile(r"\bheat(?:ed)?\s+to\s+(\d+(?:\.\d+)?)\s*°?\s*C\b", re.I),
     lambda m: [f"set heating device to {m.group(1)} °C",
                "monitor temperature until set point reached"]),
    (re.compile(r"\bmaintain(?:ed)?\s+at\s+(\d+(?:\.\d+)?)\s*°?\s*C\s+for\s+([\dhmsec\s]+)", re.I),
     lambda m: [f"hold temperature at {m.group(1)} °C for {m.group(2).strip()}"]),
    (re.compile(r"\bcool(?:ed)?\s+to\s+(\d+(?:\.\d+)?)\s*°?\s*C\b", re.I),
     lambda m: [f"cool vessel to {m.group(1)} °C"]),
    (re.compile(r"\bfilte?r(?:ed|ation)?\b", re.I),
     lambda m: ["assemble filtration apparatus",
                "pass mixture through filter",
                "collect specified fraction (filtrate/solid)"]),
    (re.compile(r"\bcentrifug(e|ed|ation)\b.*?(?P<rpm>\d+.*?rpm)?(?P<time>\d+.*?min)?", re.I),
     lambda m: ["load tubes into centrifuge",
                (f"set speed {m.group('rpm').strip()}" if m.group('rpm') else "set speed as specified"),
                (f"run for {m.group('time').strip()}" if m.group('time') else "run for specified time"),
                "separate supernatant and pellet as specified"]),
    (re.compile(r"\b(purge|degass?|bubble)\s+(with\s+)?(n2|nitrogen|argon|ar)\b", re.I),
     lambda m: ["connect inert gas line to vessel",
                "open gas flow to purge headspace",
                "maintain flow for specified duration"]),
    (re.compile(r"\b(dry|evaporate)\b.*\b(vacuum|vac)\b", re.I),
     lambda m: ["place sample in vacuum chamber",
                "apply vacuum until solvent removed or mass constant"]),
    (re.compile(r"\badjust\s+pH\s+to\s+(\d+(?:\.\d+)?)", re.I),
     lambda m: [f"measure solution pH",
                f"add acid/base to reach pH {m.group(1)}",
                "verify pH is stable"]),
    (re.compile(r"\bsonicat(e|ed|ion)\b", re.I),
     lambda m: ["place container in ultrasonic bath",
                "run sonication for specified time / power"]),
    (re.compile(r"\bwash(?:ed)?\s+with\s+(.*)", re.I),
     lambda m: [f"wash solid with {m.group(1).strip()}",
                "discard washings or combine as specified"]),
]
def _expand_procedure_line(line: str) -> Action:
    for rx, fn in ACTION_RULES:
        m = rx.search(line)
        if m: return fn(m)
    return [line]

# ---------- robot-mode helpers ----------
def _strip_reasoning_fragments(s: str) -> str:
    s = _STRIP_CIT_RX.sub("", s)
    s = re.sub(r"\((?:[^)]{0,80})\)", lambda m: ("" if _REASONING_MARKERS.search(m.group(0)) else m.group(0)), s)
    s = _TRAIL_REASON_RX.sub("", s)
    s = re.sub(r"[:–—-]\s*(?=(because|since|so that|therefore|thus|hence)\b).*", "", s, flags=re.I)
    s = re.sub(r"\b(because|since|so that|therefore|thus|hence|rationale|justif(?:y|ication)|note:|ensuring).*$", "", s, flags=re.I)
    return s.strip()

def _is_meta_line(s: str) -> bool:
    return bool(_META_RX.search(s))

def _normalize_prepare_line(s: str) -> str:
    s2 = re.sub(r"^\s*in a separate (container|vessel),\s*", "", s, flags=re.I)
    s2 = re.sub(r"\bprepare\s+a\b", "prepare", s2, flags=re.I)
    return s2.strip()

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
    """Remove repeat setup lines appearing multiple times non-adjacently."""
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

def _expand_prepare_solution(line: str) -> Optional[List[str]]:
    m = _PREPARE_SOLUTION_RX.search(line)
    if not m:
        return None
    amt, aunit = m.group("amt"), m.group("aunit")
    chem = re.sub(r"\s+", " ", (m.group("chem") or "").strip())
    vol, vunit = m.group("vol"), m.group("vunit")
    solv = re.sub(r"\s+", " ", (m.group("solv") or "").strip())
    # Atomic steps for a robot
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

def _assign_targets(steps: List[str], vessels: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Stateful assignment: maintain current vessel; open new on 'in a <vessel>'."""
    if not vessels:
        vessels = [{"id": "V1", "type": "flask", "description": "flask"}]
    current = vessels[0]["id"]  # default V1
    out: List[Dict[str, str]] = []
    for s in steps:
        # Create new vessel if step introduces one (e.g., "in a 50 mL centrifuge tube")
        m = _SEPARATE_VESSEL_RX.search(s)
        if m:
            vol = (m.group("vol") or "").strip()
            vu  = (m.group("vu")  or "").strip()
            typ = (m.group("typ") or "").strip().lower()
            desc = (f"{vol} {vu} " if vol and vu else "") + typ
            desc = re.sub(r"\s+", " ", desc).strip()
            current = _ensure_vessel(vessels, desc)

        if re.search(r"\bcentrifug", s, re.I):
            tube = next((v for v in vessels if "tube" in v["description"].lower()), None)
            if tube:
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

    # Expand to low-level actions first (low-level verb rules)
    expanded: List[str] = []
    for line in procedure_lines:
        expanded.extend(_expand_procedure_line(line))

    vessels = _collect_vessels_from_hardware(hardware_lines)

    # --- robot mode cleanup + vessel labeling / targeting ---
    final_actions = expanded
    if robot:
        tmp: List[str] = []
        for s in expanded:
            if _is_meta_line(s):
                continue
            s = _normalize_prepare_line(s)
            # Split “prepare X g Y in Z mL solvent” into atomic steps (if present)
            atomic = _expand_prepare_solution(s)
            if atomic:
                tmp.extend(atomic)
                continue
            # Otherwise clean reasoning
            s2 = _strip_reasoning_fragments(s)
            if s2:
                tmp.append(s2)

        # Remove adjacent duplicates, then global setup duplicates
        tmp = _dedupe_adjacent(tmp)
        final_actions = _global_dedupe_setup(tmp)

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
        out["robot"] = {"cleaned": True, "vessel_labels": True, "atomic_prepare": True}

    if not (out["reagents"] or out["procedure"] or out.get("hardware")):
        raise ParserError("Could not recognize any sections or steps.")

    return out

if __name__ == "__main__":
    import sys
    print(json.dumps(convert_to_json(open(sys.argv[1], "r", encoding="utf-8").read(), robot=True),
                     indent=2, ensure_ascii=False))
