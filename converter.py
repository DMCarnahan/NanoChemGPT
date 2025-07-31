from __future__ import annotations
import json, re, math, textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Tuple

from chemcrow.tools.experimental import ExperimentalTools
from chemcrow.agent import ChemCrow

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
        (?P<name>hardware(?:\s*&\s*glassware)?|glassware|materials|reagents|procedure|steps|method)
        (?:\*\*)?
        \s*:?\s*$
    """, re.I | re.VERBOSE)

_LIST_BULLET = re.compile(r"^\s*(?:[-*•–—]\s+|\d+[\.\)]\s+)")
_AMOUNT_RX = re.compile(r"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg|µl|μl|ul|mL|ml|L|l)\b", re.I)
_CONC_RX   = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mM|M|%(?:\s*w\/v|\s*v\/v)?)\b", re.I)
_VOL_RX    = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<vunit>mL|L)\b", re.I)

_REASONING_MARKERS = re.compile(r"\b(because|since|so that|therefore|thus|hence|rationale|justif(?:y|ication)|ensuring)\b", re.I)
_META_RX = re.compile(r"^\s*(this (protocol|procedure)|these steps|the following (procedure|protocol)|which can be|overview|background|this will serve|this step is crucial|ensure stability)\b", re.I)
_STRIP_CIT_RX = re.compile(r"\s*\[(?:CTX|GEN|\d+)\]\s*$", re.I)

_TRAIL_REASON_RX = re.compile(r"\s*(?:[,;\.\-–—]\s*)?(?:to\s+(?:ensure|facilitate|promote|allow|prevent|remove|minimi[sz]e|improve|enhance)\b.*|ensuring\b.*)$", re.I)
_SECOND_SENTENCE_REASON_RX = re.compile(r"\.\s*(?:This|These|It is|Note that)\b[^.]{0,200}?(?:crucial|important|helps?|ensur(?:e|es)|to\s+(?:prevent|remove|minimi[sz]e|enhance)).*$", re.I)

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

def _map_p2a_action_to_schema(act: dict) -> dict:
    """
    Map a Paragraph2Actions action dict to NanoChemGPT atomic step schema.
    """
    # Robust mapping with field normalization
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
    # Map vessel
    vessel = act.get("vessel")
    if vessel:
        step["vessel"] = vessel
    # Map temperature
    temp = act.get("temperature")
    if temp:
        step["temperature"] = temp
    # Map duration
    duration = act.get("duration")
    if duration:
        step["duration"] = duration
    # Map additional fields if present
    for key in ("dropwise", "atmosphere", "pressure", "rpm", "rate", "ph", "yield"):
        if act.get(key) is not None:
            step[key] = act[key]
    # Remove empty fields
    return {k: v for k, v in step.items() if v not in (None, "", [], {})}

def extract_atomic_steps_with_p2a(procedure_text: str) -> list:
    """
    Use Paragraph2Actions to extract atomic steps from a procedure paragraph.
    Returns a list of dicts, each representing an atomic action.
    """
    if not _P2A_MODEL:
        raise ParserError("Paragraph2Actions is not installed. Run 'pip install paragraph2actions'.")
    actions = _P2A_MODEL.predict(procedure_text)
    atomic_steps = []
    for act in actions:
        mapped = _map_p2a_action_to_schema(act)
        atomic_steps.append(mapped)
    return atomic_steps

def extract_atomic_steps_with_chemcrow(procedure_text: str) -> list:
    """
    Use ChemCrow to extract atomic steps from a procedure paragraph.
    Returns a list of dicts, each representing an atomic action.
    """
    # Initialize ChemCrow agent (do this once in production for efficiency)
    agent = ChemCrow(tools=[ExperimentalTools()])
    # Prompt ChemCrow to convert the procedure to atomic steps
    prompt = (
        "Convert the following chemical procedure into a list of atomic, robot-executable steps. "
        "Each step should be a JSON object with fields like 'action', 'reagents', 'amount', 'vessel', 'temperature', 'duration', etc. "
        "Return a JSON array of steps. Procedure:\n"
        f"{procedure_text}"
    )
    response = agent(prompt)
    # Try to extract the JSON from the response
    import json
    try:
        # ChemCrow may return text with a code block or extra text, so extract JSON robustly
        import re
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            steps = json.loads(match.group(0))
        else:
            steps = json.loads(response)
        return steps
    except Exception as e:
        # Fallback: return the raw response as a single step
        return [{"action": "raw_chemcrow_response", "details": response}]

def convert_to_json(raw: str, robot: bool = False) -> Dict[str, object]:
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")
    raw = textwrap.dedent(raw).strip()

    # --- Section extraction logic (simplified for brevity) ---
    # This is a placeholder; use your actual section extraction logic
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
    if robot and "procedure" in sections:
        procedure_text = "\n".join(sections["procedure"])
        try:
            procedure_structured = extract_atomic_steps_with_chemcrow(procedure_text)
        except Exception as e:
            procedure_structured = [{"action": "error", "details": str(e)}]
    else:
        # fallback logic
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
    print(json.dumps(convert_to_json(txt, robot=True), indent=2, ensure_ascii=False))