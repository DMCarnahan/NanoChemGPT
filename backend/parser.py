from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

__all__ = ["convert_to_json", "ParserError"]

SCHEMA_VERSION = "1.1" 


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
    return [_bullet_rx.match(l).group(1).strip() for l in block.splitlines() if _bullet_rx.match(l)]

@dataclass
class Reagent:
    description: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    concentration: Optional[str] = None
    final_volume_mL: Optional[float] = None

    def asdict(self):
        return asdict(self)


def _parse_reagent(line: str) -> Reagent:
    amt = _amount_rx.search(line)
    conc = _conc_rx.search(line)
    vol = _volume_rx.search(line)
    return Reagent(
        description=line,
        amount=float(amt.group("qty")) if amt else None,
        unit=amt.group("unit") if amt else None,
        concentration=conc.group(0) if conc else None,
        final_volume_mL=float(vol.group("val")) if vol else None,
    )

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
        "title": raw.split("\n", 1)[0].lstrip("# ")[:120],
        "reagents": [],
        "procedure": [],
        "characterization": [],
        "storage": "",
    }

    for header, body in sections.items():
        key = _slug(header)
        bullets = _bullets(body)

        if any(k in key for k in ("reagent", "material")):
            out["reagents"] = [_parse_reagent(b).asdict() for b in bullets]
        elif "procedure" in key or "synthesis" in key:
            out["procedure"] = bullets
        elif "characterization" in key:
            out["characterization"] = bullets
        elif "storage" in key:
            out["storage"] = bullets[0] if bullets else body.strip()
        else:
            out[key] = bullets or body.strip()

    if not out["reagents"] or not out["procedure"]:
        raise ParserError("Missing required 'Materials'/'Reagents' or 'Procedure' sections.")

    return out

# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json as _j
    print(_j.dumps(convert_to_json(open(sys.argv[1]).read()), indent=2, ensure_ascii=False))
