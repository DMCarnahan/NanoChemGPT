from __future__ import annotations

import json, re, textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Tuple

__all__ = ["convert_to_json", "ParserError"]
SCHEMA_VERSION = "1.2"

class ParserError(ValueError):
    pass

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_heading_rx = re.compile(r"^\s*\d+\.\s+\*\*(.+?)\*\*:", re.M)
_bullet_rx  = re.compile(r"^\s*[\-–]\s+(.*)")
_amount_rx  = re.compile(r"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg|µl|μl|ml|mL|l|L)\b", re.I)
_conc_rx    = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mM|M|%\s*w\/v|%\s*v\/v)", re.I)
_volume_rx  = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*mL", re.I)

def _slug(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_")

def _split_sections(text: str) -> Dict[str, str]:
    parts = _heading_rx.split(text)
    headers, bodies = parts[1::2], parts[2::2]
    return {h.strip(): b.strip() for h, b in zip(headers, bodies)}

def _bullets(block: str) -> List[str]:
    return [_bullet_rx.match(l).group(1).strip()
            for l in block.splitlines() if _bullet_rx.match(l)]

@dataclass
class Reagent:
    description: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    concentration: Optional[str] = None
    final_volume_mL: Optional[float] = None
    def asdict(self): return asdict(self)

def _parse_reagent(line: str) -> Reagent:
    amt  = _amount_rx.search(line)
    conc = _conc_rx.search(line)
    vol  = _volume_rx.search(line)
    return Reagent(
        description=line,
        amount=float(amt.group("qty")) if amt else None,
        unit=amt.group("unit") if amt else None,
        concentration=conc.group(0) if conc else None,
        final_volume_mL=float(vol.group("val")) if vol else None,
    )

# ---------------------------------------------------------------------------
# Procedure expansion rules
# ---------------------------------------------------------------------------
Action = List[str]
RuleFn = Callable[[re.Match], Action]

ACTION_RULES: List[Tuple[re.Pattern, RuleFn]] = [
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
    # Heat to T °C
    (re.compile(r"\bheat(?:ed)?\s+to\s+(\d+(?:\.\d+)?)\s*°?\s*C", re.I),
     lambda m: [f"set heating device to {m.group(1)} °C",
                "monitor temperature until set point reached"]),
    # Maintain at T for time
    (re.compile(r"\bmaintain(?:ed)?\s+at\s+(\d+(?:\.\d+)?)\s*°?\s*C\s+for\s+([\dhmsec\s]+)", re.I),
     lambda m: [f"hold temperature at {m.group(1)} °C for {m.group(2).strip()}"]),
    # Cool to T °C
    (re.compile(r"\bcool(?:ed)?\s+to\s+(\d+(?:\.\d+)?)\s*°?\s*C", re.I),
     lambda m: [f"cool vessel to {m.group(1)} °C"]),
    # Filter / Filtration
    (re.compile(r"\bfilte?r(?:ed|ation)?\b", re.I),
     lambda m: ["assemble filtration apparatus",
                "pass mixture through filter",
                "collect specified fraction (filtrate/solid)"]),
    # Centrifuge
    (re.compile(r"\bcentrifug(e|ed|ation)\b.*?(?P<rpm>\d+.*?rpm)?(?P<time>\d+.*?min)?", re.I),
     lambda m: ["load tubes into centrifuge",
                f"set speed {m.group('rpm').strip()}" if m.group('rpm') else "set speed as specified",
                f"run for {m.group('time').strip()}" if m.group('time') else "run for specified time",
                "separate supernatant and pellet as specified"]),
    # Purge with gas
    (re.compile(r"\b(purge|degass?|bubble)\s+(with\s+)?(n2|nitrogen|argon|ar)\b", re.I),
     lambda m: ["connect inert gas line to vessel",
                "open gas flow to purge headspace",
                "maintain flow for specified duration"]),
    # Vacuum dry
    (re.compile(r"\b(dry|evaporate)\b.*\b(vacuum|vac)\b", re.I),
     lambda m: ["place sample in vacuum chamber",
                "apply vacuum until solvent removed / mass constant"]),
    # pH adjust
    (re.compile(r"\badjust\s+pH\s+to\s+(\d+(?:\.\d+)?)", re.I),
     lambda m: [f"measure solution pH",
                f"add acid/base to reach pH {m.group(1)}",
                "verify pH is stable"]),
    # Fallback
]

def _expand_procedure_line(line: str) -> Action:
    for rx, fn in ACTION_RULES:
        m = rx.search(line)
        if m:
            return fn(m)
    return [line]  # keep original if nothing matched

# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------
def convert_to_json(raw: str) -> Dict[str, object]:
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")

    raw = textwrap.dedent(raw).strip()
    sections = _split_sections(raw)
    if not sections:
        raise ParserError("No numbered **Header** sections found.")

    out: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "title"         : raw.split("\n", 1)[0].lstrip("# ")[:120],
        "reagents"      : [],
        "procedure"     : [],
        "hardware"      : [],
        "characterization": [],  # legacy
        "storage"       : "",
    }

    for header, body in sections.items():
        key     = _slug(header)
        bullets = _bullets(body)

        if any(k in key for k in ("reagent", "material")):
            out["reagents"] = [_parse_reagent(b).asdict() for b in bullets]

        elif "procedure" in key or "synthesis" in key:
            for b in bullets:
                out["procedure"].extend(_expand_procedure_line(b))

        elif "hardware" in key:
            out["hardware"] = bullets

        elif "characterization" in key:
            out["characterization"] = bullets

        elif "storage" in key:
            out["storage"] = bullets[0] if bullets else body.strip()

        else:
            out[key] = bullets or body.strip()

    if not out["reagents"] or not out["procedure"]:
        raise ParserError("Missing required 'Materials'/'Reagents' or 'Procedure' sections.")

    return out

if __name__ == "__main__":
    import sys
    print(json.dumps(convert_to_json(open(sys.argv[1]).read()), indent=2, ensure_ascii=False))
