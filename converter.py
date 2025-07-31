from __future__ import annotations
import json, re, textwrap
from typing import Dict, List, Tuple

__all__ = ["convert_to_json", "ParserError"]
SCHEMA_VERSION = "1.8.2"

class ParserError(ValueError):
    pass

# --- Paragraph2Actions integration (optional) ---
try:
    from paragraph2actions import Paragraph2Actions
    _P2A_MODEL = Paragraph2Actions.from_pretrained()
except ImportError:
    _P2A_MODEL = None

_HEADING_LINE = re.compile(
    r"""^\s*
        (?:\d+[\.\)]\s*)?
        (?:\*\*)?
        (?P<name>hardware(?:\s*&\s*glassware)?|glassware|materials|reagents|procedure|steps|method|title)
        (?:\*\*)?
        \s*:?\s*$
    """, re.I | re.VERBOSE)

_LIST_BULLET = re.compile(r"^\s*(?:[-*•–—]\s+|\d+[\.\)]\s+)")

def _map_p2a_action_to_schema(act: dict) -> dict:
    """Map a Paragraph2Actions action dict to NanoChemGPT atomic step schema."""
    op = act.get("operation", "").lower()
    step = {
        "action": op if op else act.get("text", "action"),
        "details": act.get("text", ""),
    }
    # Map reagents/solvents
    reagents = act.get("reagents") or []
    solvents = act.get("solvents") or []
    if reagents:
        step["reagents"] = reagents
    if solvents:
        step["solvents"] = solvents
    # Map vessel, temperature, duration, and other fields
    for key in ("vessel", "temperature", "duration", "dropwise", "atmosphere", "pressure", "rpm", "rate", "ph", "yield"):
        if act.get(key) is not None:
            step[key] = act[key]
    # Remove empty fields
    return {k: v for k, v in step.items() if v not in (None, "", [], {})}

def extract_atomic_steps_with_p2a(procedure_text: str) -> list:
    """Use Paragraph2Actions to extract atomic steps from a procedure paragraph."""
    if not _P2A_MODEL:
        raise ParserError("Paragraph2Actions is not installed. Run 'pip install paragraph2actions'.")
    actions = _P2A_MODEL.predict(procedure_text)
    return [_map_p2a_action_to_schema(act) for act in actions]

def extract_atomic_steps_with_chemcrow(procedure_text: str) -> list:
    """
    Use ChemCrow to extract atomic steps from a procedure paragraph.
    Returns a list of dicts, each representing an atomic action.
    """
    # Import ChemCrow only here to avoid global state issues
    from chemcrow.tools.experimental import ExperimentalTools
    from chemcrow.agent import ChemCrow
    agent = ChemCrow(tools=[ExperimentalTools()])
    prompt = (
        "Convert the following chemical procedure into a list of atomic, robot-executable steps. "
        "Each step should be a JSON object with fields like 'action', 'reagents', 'amount', 'vessel', 'temperature', 'duration', etc. "
        "Return a JSON array of steps. Procedure:\n"
        f"{procedure_text}"
    )
    response = agent(prompt)
    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            steps = json.loads(match.group(0))
        else:
            steps = json.loads(response)
        return steps
    except Exception:
        return [{"action": "raw_chemcrow_response", "details": response}]

def convert_to_json(raw: str, robot: bool = False, use_chemcrow: bool = True) -> Dict[str, object]:
    """
    Convert a raw protocol string to structured JSON.
    If robot=True, will use ChemCrow (default) or Paragraph2Actions for atomic steps.
    """
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")
    raw = textwrap.dedent(raw).strip()

    # --- Section extraction ---
    sections = {}
    current = None
    for line in raw.splitlines():
        m = _HEADING_LINE.match(line)
        if m:
            current = m.group("name").lower()
            sections[current] = []
            continue
        if current:
            if _LIST_BULLET.match(line) or line.strip():
                sections[current].append(line.strip())

    # --- Use ChemCrow or Paragraph2Actions for atomic steps if robot mode and procedure present ---
    procedure_structured = []
    if robot and "procedure" in sections:
        procedure_text = "\n".join(sections["procedure"])
        try:
            if use_chemcrow:
                procedure_structured = extract_atomic_steps_with_chemcrow(procedure_text)
            else:
                procedure_structured = extract_atomic_steps_with_p2a(procedure_text)
        except Exception as e:
            procedure_structured = [{"action": "error", "details": str(e)}]
    else:
        # fallback logic: just split lines
        procedure_structured = [{"action": l} for l in sections.get("procedure", [])]

    # --- Compose output ---
    result = {
        "schema_version": SCHEMA_VERSION,
        "title": sections.get("title", ["SynthesisProtocol"])[0] if "title" in sections else "SynthesisProtocol",
        "hardware": sections.get("hardware", []),
        "materials": sections.get("materials", []),
        "procedure": sections.get("procedure", []),
        "procedure_structured": procedure_structured,
        # Add other fields as needed
    }
    return result

if __name__ == "__main__":
    import sys
    txt = open(sys.argv[1], "r", encoding="utf-8").read()
    # Default: use ChemCrow for atomic steps if robot=True
    print(json.dumps(convert_to_json(txt, robot=True), indent=2, ensure_ascii=False))