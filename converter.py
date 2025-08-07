"""
converter.py – Convert NanoChemGPT answers into structured JSON
----------------------------------------------------------------

This version:
  • Normalises section headings (“Hardware”, “Materials”, “Procedure”, etc.).
  • Cleans bullets and trailing [CTX]/[GEN]/[n] tags from hardware/materials.
  • Enriches materials (stub implementation).
  • Calls an OpenAI function (‘add_step’) to extract atomic steps with
    explicit fields (action, time, temperature, vessel, reagents, solvents).
  • Maps each vessel description to an identifier (V1, V2, …) based on the
    order of the hardware list; this makes the JSON robot-friendly.
"""

from __future__ import annotations
import json
import os
import re
import textwrap
from typing import Any, Dict, List

try:
    import openai
except ImportError:
    openai = None  # allow offline use

_CLIENT = None
if openai and os.getenv("OPENAI_API_KEY"):
    _CLIENT = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# 1. Regex helpers: strip bullets & tags, detect section headings
# ---------------------------------------------------------------------------
_HEADING = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)?(?:\*\*)?(?P<name>hardware(?:\s*&\s*glassware)?|glassware|materials|reagents|procedure|steps|method|title)(?:\*\*)?\s*:?\s*$",
    re.I
)
_BULLET = re.compile(r"^\s*(?:[-*•–—]\s+|\d+[\.\)]\s+)")
_TAGS   = re.compile(r"\s*\[(?:CTX|DB|PARSED|GEN|\d+)]\s*[.。;:,-]?\s*$")

def _clean_line(line: str) -> str:
    """Remove leading bullet/number and trailing provenance tags."""
    return _TAGS.sub("", _BULLET.sub("", line)).strip()

# ---------------------------------------------------------------------------
# 2. Split answer into named sections
# ---------------------------------------------------------------------------
def _split_sections(raw: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: str | None = None
    for line in raw.splitlines():
        m = _HEADING.match(line)
        if m:
            current = m.group("name").lower()
            sections[current] = []
            continue
        if current:
            stripped = line.strip()
            if stripped:
                sections[current].append(stripped)
    return sections

# ---------------------------------------------------------------------------
# 3. Enrich materials (placeholder)
# ---------------------------------------------------------------------------
def enrich_materials(lines: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for ln in lines:
        if " as " in ln:
            name, notes = ln.split(" as ", 1)
            out.append({"name": name.strip(), "notes": _clean_line(notes)})
        else:
            out.append({"name": ln})
    return out

# ---------------------------------------------------------------------------
# 4. Function-calling schema for atomic steps
# ---------------------------------------------------------------------------
_FN_SCHEMA = {
    "name": "add_step",
    "description": (
        "Add one atomic operation in a chemical synthesis step. "
        "For each call, extract explicit fields when present: "
        "action (a verb), time (duration), temperature, vessel, reagents, solvents, and optional notes. "
        "Omit any field that is not present."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action":      {"type": "string", "description": "verb for the step"},
            "time":        {"type": "string", "description": "duration (e.g. '1 h')"},
            "temperature": {"type": "string", "description": "temperature (e.g. '70 °C')"},
            "vessel":      {"type": "string", "description": "reaction vessel (e.g. '250 mL beaker')"},
            "reagents":    {"type": "array", "items": {"type": "string"}},
            "solvents":    {"type": "array", "items": {"type": "string"}},
            "notes":       {"type": "string", "description": "additional information"},
        },
        "required": ["action"]
    },
}

_SYSTEM_PROMPT = textwrap.dedent("""
  You are a chemistry protocol parser.
  For EACH individual sentence below call *add_step* exactly once.
  Do NOT combine multiple operations in one call.
  Extract these fields when present:
    • action  • time  • temperature  • vessel  • reagents  • solvents  • notes
  Omit any field that is absent.
""").strip()

def gpt_steps(paragraphs: List[str], model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    """Call the OpenAI model to extract atomic steps with explicit fields."""
    if not _CLIENT or not paragraphs:
        return []
    msgs = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for line in paragraphs:
        # split on '.', ';', then strip
        for sent in re.split(r"[.;](?=\s|$)", line):
            sent = sent.strip()
            if sent:
                msgs.append({"role": "user", "content": sent})
    # One user message per paragraph
    for p in paragraphs:
        clean = re.sub(r"^\s*\d+[.)]\s*", "", p).strip()
        if clean:
            msgs.append({"role": "user", "content": clean})
    resp = _CLIENT.chat.completions.create(
        model=model,
        temperature=0.1,
        tools=[{"type": "function", "function": _FN_SCHEMA}],
        tool_choice={"type": "function", "function": {"name": "add_step"}},
        messages=msgs,
    )
    steps: List[Dict[str, Any]] = []
    for ch in resp.choices:
        # New API returns tool role messages
        if getattr(ch.message, "role", None) == "tool":
            steps.append(json.loads(ch.message.content))
        elif getattr(ch.message, "tool_calls", None):
            for tc in ch.message.tool_calls:
                steps.append(json.loads(tc.function.arguments))
    return steps

# ---------------------------------------------------------------------------
# 5. Vessel mapping: assign V1, V2,… based on hardware list
# ---------------------------------------------------------------------------
def _vessel_map(hardware: List[str]) -> Dict[str, str]:
    return {hw.lower(): f"V{i+1}" for i, hw in enumerate(hardware)}

def _assign_vessels(steps: List[Dict[str, Any]], vmap: Dict[str, str]) -> List[Dict[str, Any]]:
    """Replace vessel descriptions with IDs (e.g. 'V1', 'V2')."""
    for s in steps:
        vessel = s.get("vessel")
        if vessel:
            low = vessel.lower()
            for key, vid in vmap.items():
                if key in low:
                    s["vessel"] = vid
                    break
    return steps

# ---------------------------------------------------------------------------
# 6. Public API: raw text → structured JSON
# ---------------------------------------------------------------------------
class ParserError(RuntimeError): ...

def convert_to_json(raw: str, *, robot: bool = False) -> Dict[str, Any]:
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")
    raw = textwrap.dedent(raw).strip()

    sections = _split_sections(raw)

    # Structured steps (call model if requested)
    procedure_structured: List[Dict[str, Any]] = []
    if sections.get("procedure"):
        if robot:
            paragraphs = [ln for ln in sections["procedure"] if ln.strip()]
            try:
                steps = gpt_steps(paragraphs, model="gpt-4o-mini")
                # Map vessel names to IDs and drop empty fields
                vmap = _vessel_map(sections.get("hardware", []))
                steps = _assign_vessels(steps, vmap)
                procedure_structured = steps
            except Exception as exc:
                procedure_structured = [{"action": "error", "notes": f"extract: {exc}"}]
        else:
            procedure_structured = [{"action": "step", "notes": ln} for ln in sections["procedure"]]

    # Materials enrichment
    materials_lines = sections.get("materials", [])
    try:
        materials_struct = enrich_materials([_clean_line(ln) for ln in materials_lines])
    except Exception as exc:
        materials_struct = [{"name": ln, "notes": f"enrich failed: {exc}"} for ln in materials_lines]

    # Title fallback
    title = sections.get("title", ["SynthesisProtocol"])
    title = title[0] if title else "SynthesisProtocol"

    return {
        "title": title,
        "hardware": sections.get("hardware", []),
        "materials": materials_lines,
        "materials_enriched": materials_struct,
        "procedure": sections.get("procedure", []),
        "procedure_structured": procedure_structured,
    }

# ---------------------------------------------------------------------------
# 7. Stand-alone CLI use
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pathlib
    ap = argparse.ArgumentParser(description="Convert answer text to JSON.")
    ap.add_argument("file", help="Path to text file containing NanoChemGPT answer")
    ap.add_argument("--robot", action="store_true", help="Use model to extract atomic steps")
    ns = ap.parse_args()
    text = pathlib.Path(ns.file).read_text(encoding="utf-8")
    print(
        json.dumps(
            convert_to_json(text, robot=ns.robot),
            indent=2,
            ensure_ascii=False,
        )
    )
