from __future__ import annotations
import json, re, textwrap
from typing import Dict, List

__all__ = ["convert_to_json", "ParserError"]
SCHEMA_VERSION = "1.8.2"

class ParserError(ValueError):
    pass

# --- Paragraph2Actions integration ---
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
    reagents = act.get("reagents") or []
    solvents = act.get("solvents") or []
    if reagents:
        step["reagents"] = reagents
    if solvents:
        step["solvents"] = solvents
    for key in ("vessel", "temperature", "duration", "dropwise", "atmosphere", "pressure", "rpm", "rate", "ph", "yield"):
        if act.get(key) is not None:
            step[key] = act[key]
    return {k: v for k, v in step.items() if v not in (None, "", [], {})}

def extract_atomic_steps_with_p2a(procedure_text: str) -> list:
    """Use Paragraph2Actions to extract atomic steps from a procedure paragraph."""
    if not _P2A_MODEL:
        raise ParserError("Paragraph2Actions is not installed. Run 'pip install paragraph2actions'.")
    actions = _P2A_MODEL.predict(procedure_text)
    return [_map_p2a_action_to_schema(act) for act in actions]

def convert_to_json(raw: str, robot: bool = False) -> Dict[str, object]:
    """
    Convert a raw protocol string to structured JSON.
    If robot=True, will use Paragraph2Actions for atomic steps.
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

    # --- Use Paragraph2Actions for atomic steps if robot mode and procedure present ---
    procedure_structured = []
    if robot and "procedure" in sections:
        procedure_text = "\n".join(sections["procedure"])
        try:
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
    }
    return result

if __name__ == "__main__":
    import sys
    txt = open(sys.argv[1], "r", encoding="utf-8").read()
    print(json.dumps(convert_to_json(txt, robot=True), indent=2, ensure_ascii=False))