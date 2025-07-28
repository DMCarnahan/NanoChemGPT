from __future__ import annotations

import json, re, textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Tuple

__all__ = ["convert_to_json", "ParserError"]
SCHEMA_VERSION = "1.2"

class ParserError(ValueError):
    pass

# --- tolerant headings ---
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
_VOL_RX    = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*mL\b", re.I)

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
        if not line.strip():
            continue
        norm = re.sub(r"[\*_`]", "", line).strip()
        m = _HEADING_LINE.match(norm)
        if m:
            current = _canon(m.group("name"))
            continue
        if current:
            buckets[current].append(raw)
        else:
            buckets.setdefault("_pre", []).append(raw)

    return buckets

def _bullets(lines: List[str]) -> List[str]:
    if not lines:
        return []
    any_bullets = any(_LIST_BULLET.match(l) for l in lines)
    items: List[str] = []
    for l in lines:
        s = l.strip()
        if not s:
            continue
        if any_bullets:
            s = _LIST_BULLET.sub("", s).strip()
        items.append(s)
    return items

# --- reagents ---
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

# --- procedure expansion ---
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
        if m:
            return fn(m)
    return [line]

# --- main ---
def convert_to_json(raw: str) -> Dict[str, object]:
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

    expanded: List[str] = []
    for line in procedure_lines:
        expanded.extend(_expand_procedure_line(line))

    out: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "title"         : raw.split("\n", 1)[0].lstrip("# ")[:120],
        "hardware"      : hardware_lines,
        "reagents"      : reagents,
        "procedure"     : expanded,
        "characterization": [],
        "storage"       : "",
    }

    if not (out["reagents"] or out["procedure"] or out["hardware"]):
        raise ParserError("Could not recognize any sections or steps.")

    return out

if __name__ == "__main__":
    import sys
    print(json.dumps(convert_to_json(open(sys.argv[1], "r", encoding="utf-8").read()),
                     indent=2, ensure_ascii=False))
