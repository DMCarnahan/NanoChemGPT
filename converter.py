"""
Low-level procedure parser that emits machine-executable steps.

Key features:
- ALWAYS emits an "action" for each step (never missing).
- Converts high-level "prepare solution" into explicit "dispense" with:
  solute, solvent, concentration (M), volume (mL/L).
- Cleans bracket tags like [GEN], [CTX], [1].
- Extracts coarse temperature (°C) and duration (minutes) when present.
- Leaves original step text in "raw" for auditing.

CLI:
    python converter.py /path/to/answer.txt /path/to/output.json
"""

from __future__ import annotations
import re, json, sys, pathlib
from typing import List, Dict, Optional

# ---------------- Utilities ----------------

TAG_RX = re.compile(r"\s*\[(?:CTX|DB|PARSED|GEN|\d+)\]\s*$")

def strip_tags(s: str) -> str:
    return TAG_RX.sub("", s).strip()

def _extract_temperature_c(t: str) -> float:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*C\b", t, re.I)
    return float(m.group(1)) if m else 0.0

def _extract_duration_minutes(t: str) -> float:
    mins = 0.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:hour|hr|hrs|h)\b", t, re.I):
        mins += float(m.group(1)) * 60
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:minute|min|mins|m)\b", t, re.I):
        mins += float(m.group(1))
    return mins

# ---------------- Action inference ----------------

_ACTION_PATTERNS = [
    (re.compile(r"\ballow(?:s|ed)?\b.*?\bto\s+(stir|react|age|settle|cool|heat|evaporate)\b", re.I), lambda m: m.group(1)),
    (re.compile(r"^\s*(?:\d+\.)?\s*(?:then\s+|and\s+)?\b"
                r"(prepare|add|stir|mix|heat|cool|centrifuge|wash|dry|filter|sonicate|degas|inject|age|reflux|quench|"
                r"dissolve|pour|transfer|grind|calcine|anneal|evaporate|precipitate|collect)\b", re.I), lambda m: m.group(1)),
    (re.compile(r"\b(prepare|add|stir|mix|heat|cool|centrifuge|wash|dry|filter|sonicate|degas|inject|age|reflux|"
                r"quench|dissolve|pour|transfer|grind|calcine|anneal|evaporate|precipitate|collect)\b", re.I), lambda m: m.group(1)),
]

_ACTION_MAP = {
    "prepare": "prepare",
    "add": "add",
    "stir": "stir",
    "mix": "mix",
    "heat": "heat",
    "cool": "cool",
    "centrifuge": "centrifuge",
    "wash": "wash",
    "dry": "dry",
    "filter": "filter",
    "sonicate": "sonicate",
    "degas": "degas",
    "inject": "inject",
    "age": "age",
    "reflux": "reflux",
    "quench": "quench",
    "dissolve": "dissolve",
    "pour": "pour",
    "transfer": "transfer",
    "grind": "grind",
    "calcine": "calcine",
    "anneal": "anneal",
    "evaporate": "evaporate",
    "precipitate": "precipitate",
    "collect": "collect",
    "react": "react",
    "settle": "settle",
}

def get_action(step_text: str) -> str:
    s = step_text.strip()
    for rx, f in _ACTION_PATTERNS:
        m = rx.search(s)
        if m:
            verb = f(m).lower()
            return _ACTION_MAP.get(verb, verb)
    return "process"

# ---------------- Solution prep → dispense ----------------
# Broadened phrase detection:
#   1) "Prepare a 0.1 M solution of X by dissolving Y in 100 mL of solvent"
#   2) "Prepare a 0.1 M X solution by dissolving Y in 100 mL of solvent"
#   3) "Dissolve Y in 100 mL of solvent to make a 0.1 M X solution"
#   4) "Add/charge Y to 100 mL of solvent to obtain a 0.1 M X solution"
#   5) "Make a 0.1 M X solution by dissolving Y in 100 mL solvent"
#   6) "Formulate a 0.1 M X solution ..."
#   7) "Compose a 0.1 M X solution ..."

_CONC_UNIT_RX = r"(?:M|m)"  # molarity only for now

def _clean_solvent_tail(solvent: str) -> str:
    solvent = strip_tags(solvent.strip().rstrip(",."))
    solvent = solvent.split(" in ")[0].strip()
    return solvent

def _mk_dispense(solute: str, solvent: str, conc: float, vol: float, vol_unit: str) -> Dict:
    solute = strip_tags(solute.strip().rstrip(",."))
    solvent = _clean_solvent_tail(solvent)
    return {
        "action": "dispense",
        "solute": solute,
        "solvent": solvent,
        "concentration": float(conc),
        "concentration_units": "M",
        "volume": float(vol),
        "volume_units": vol_unit,
        "identity": solute,
        "reagents": [solute, solvent],
    }

def parse_solution_prep(step_text: str) -> Optional[Dict]:
    s = step_text.strip().rstrip(".")

    patterns = [
        # 1) Prepare a 0.1 M solution of X by dissolving Y in 100 mL of solvent
        re.compile(
            rf"""prepare\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+solution\s+of\s+.+?\s+
                by\s+dissolving\s+(?:an\s+appropriate\s+amount\s+of\s+)?
                (?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s*(?:in\b|$)""",
            re.I | re.X,
        ),
        # 2) Prepare a 0.1 M X solution by dissolving Y in 100 mL of solvent
        re.compile(
            rf"""prepare\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+(?P<xname>.+?)\s+solution\s+
                by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s*(?:in\b|$)""",
            re.I | re.X,
        ),
        # 3) Dissolve Y in 100 mL of solvent to make a 0.1 M X solution
        re.compile(
            rf"""dissolv\w*\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s+
                to\s+(?:make|form|yield|obtain)\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+.+?\s+solution""",
            re.I | re.X,
        ),
        # 4) Add/charge Y to 100 mL of solvent to obtain a 0.1 M X solution
        re.compile(
            rf"""(?:add|charge)\s+(?P<solute>.+?)\s+to\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s+
                to\s+(?:make|form|yield|obtain)\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+.+?\s+solution""",
            re.I | re.X,
        ),
        # 5) Make a 0.1 M X solution by dissolving Y in 100 mL solvent
        re.compile(
            rf"""(?:make|formulate|compose)\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+.+?\s+solution\s+
                by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\b""",
            re.I | re.X,
        ),
        # 6) Make X (0.1 M) by dissolving Y in 100 mL of solvent
        re.compile(
            rf"""(?:make|formulate|compose)\s+.+?\(\s*([\d\.]+)\s*({_CONC_UNIT_RX})\s*\)\s+
                by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\b""",
            re.I | re.X,
        ),
    ]

    for rx in patterns:
        m = rx.search(s)
        if m:
            conc = float(m.group(1))
            solute = (m.groupdict().get("solute") or "").strip()
            vol = float(m.group("vol"))
            vunit = m.group("vunit")
            solvent = m.group("solvent")
            return _mk_dispense(solute, solvent, conc, vol, vunit)

    return None

# ---------------- Record construction ----------------

def build_record_from_step(step_text: str) -> Dict:
    # 1) Try solution-prep → dispense
    sol_prep = parse_solution_prep(step_text)
    if sol_prep:
        sol_prep.setdefault("temperature", 0.0)
        sol_prep.setdefault("duration", 0.0)
        sol_prep["raw"] = step_text.strip()
        return sol_prep

    # 2) Fallback: infer action + minimal fields
    action = get_action(step_text)
    return {
        "action": action,
        "identity": "",
        "reagents": [],
        "solvent": "",
        "amount": 0,
        "units": "",
        "temperature": _extract_temperature_c(step_text),
        "duration": _extract_duration_minutes(step_text),
        "raw": step_text.strip(),
    }

# ---------------- Markdown procedure extraction ----------------

def parse_procedure_blocks(markdown_text: str) -> List[str]:
    lines = markdown_text.splitlines()
    in_proc = False
    steps: List[str] = []
    step_buf: List[str] = []
    for line in lines:
        if re.match(r"\s*3\.\s*\*\*Procedure\*\*:", line):
            in_proc = True
            continue
        if in_proc:
            if re.match(r"\s*\d+\.\s", line):
                if step_buf:
                    steps.append(" ".join(step_buf).strip())
                    step_buf = []
                step_buf.append(re.sub(r"^\s*\d+\.\s*", "", line).strip())
            else:
                step_buf.append(line.strip())
    if step_buf:
        steps.append(" ".join(step_buf).strip())
    # Clean trailing tags
    steps = [strip_tags(s) for s in steps if s.strip()]
    return steps

# ---------------- Public API ----------------

def convert_text(text: str) -> List[Dict]:
    steps = parse_procedure_blocks(text)
    return [build_record_from_step(s) for s in steps]

# ---------------- CLI ----------------

def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print("Usage: python converter.py <input_txt> <output_json>")
        return 2
    in_path = pathlib.Path(argv[1])
    out_path = pathlib.Path(argv[2])
    text = in_path.read_text(encoding="utf-8")
    records = convert_text(text)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
