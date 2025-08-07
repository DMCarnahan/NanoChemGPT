"""
Turn a formatted “NanoChemGPT” answer into explicit JSON
-----------------------------------------------------------------------
Key changes vs. previous version
• Normalises “Hardware & Glassware”, “Hardware”, or “Glassware” → sections["hardware"]
• Cleans leading bullets and trailing [n] citations from materials lines
• Modern OpenAI tool-call: no forced priming; parses messages with role="tool"
"""

from __future__ import annotations
import os, re, json, textwrap, itertools
from typing import List, Dict, Any

import openai

# ════════════════════  0. OpenAI client  ═══════════════════════════════════ #
_openai_api_key = os.getenv("OPENAI_API_KEY") or ""
_client = openai.OpenAI(api_key=_openai_api_key) if _openai_api_key else None

# ════════════════════  1. Helpers  ════════════════════════════════════════ #
_HDR_RX   = re.compile(r"^\s*##\s*(.+?)\s*$", re.M)
_SUB_RX   = re.compile(r"^\s*\d+\.\s*\*\*(?P<name>[^*]+?)\*\*[:\s]*$", re.M)

_BULLET_RX = re.compile(r"^\s*[-*•–—]\s*")
_CITE_RX   = re.compile(r"\s*\[\d+\]\s*$")

def _clean_item(line: str) -> str:
    """Strip bullet symbols and trailing [n] inline citations."""
    return _CITE_RX.sub("", _BULLET_RX.sub("", line)).strip()

def _split_sections(txt: str) -> Dict[str, List[str]]:
    """
    Return {'hardware': [...], 'materials': [...], 'procedure': [...], 'other': [...]}
    Keys are lowered; unrecognised headings go into 'other'.
    """
    sections: Dict[str, List[str]] = {"hardware": [], "materials": [], "procedure": [], "other": []}
    current = "other"

    for line in txt.splitlines():
        m = _SUB_RX.match(line)
        if m:
            name = m.group("name").lower().strip()
            # normalise heading names
            if name.startswith("hardware") or name in ("glassware",):
                current = "hardware"
            elif name.startswith("material"):
                current = "materials"
            elif name.startswith("procedure"):
                current = "procedure"
            else:
                current = "other"
            continue
        sections.setdefault(current, []).append(line.rstrip())

    # strip blank lines from each section list
    for k in list(sections):
        sections[k] = [l for l in sections[k] if l.strip()]
    return sections

# ════════════════════  2. Materials enrichment (toy example)  ═════════════ #
def enrich_materials(lines: List[str]) -> List[Dict[str, str]]:
    """Very naive split into 'name' & 'notes'."""
    out: List[Dict[str, str]] = []
    for ln in lines:
        parts = ln.split(" as ", 1)
        if len(parts) == 2:
            out.append({"name": parts[0].strip(), "notes": parts[1].strip()})
        else:
            out.append({"name": ln})
    return out

# ════════════════════  3. GPT function-calling to get atomic steps  ═══════ #
_fn_schema = {
    "name": "add_step",
    "description": "Add a fully explicit atomic step to the procedure",
    "parameters": {
        "type": "object",
        "properties": {
            "action":   {"type": "string", "description": "verb phrase, e.g. 'heat', 'add', 'filter'"},
            "details":  {"type": "string", "description": "full description incl. temperature, time"},
            "reagents": {"type": "array", "items": {"type": "string"}, "description": "chemical names"},
            "solvents": {"type": "array", "items": {"type": "string"}, "description": "solvent names"}
        },
        "required": ["action", "details"]
    }
}

_SYSTEM_STEPS = textwrap.dedent("""
    You are a chemistry protocol parser.
    Break the PROCEDURE text into explicit atomic steps, calling the add_step
    tool once for each distinct action. 8–20 steps is typical.
""").strip()

def gpt_steps(paragraphs: List[str], model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    if _client is None:
        return []

    msgs = [
        {"role": "system", "content": _SYSTEM_STEPS},
        {"role": "user", "content": "PROCEDURE:\n" + "\n".join(paragraphs)}
    ]

    resp = _client.chat.completions.create(
        model=model,
        temperature=0,
        tools=[{"type": "function", "function": _fn_schema}],
        messages=msgs,
    )

    steps: List[Dict[str, Any]] = []
    for ch in resp.choices:
        if ch.message.role == "tool":
            steps.append(json.loads(ch.message.content))
        elif ch.message.tool_calls:
            for tc in ch.message.tool_calls:
                steps.append(json.loads(tc.function.arguments))
    return steps

# ════════════════════  4. Public API  ═════════════════════════════════════ #
class ParserError(RuntimeError): ...

def convert_to_json(answer_text: str, *, robot: bool = False) -> Dict[str, Any]:
    """
    Convert the model's answer into rich JSON suitable for execution by a robot.
    """
    sections = _split_sections(answer_text)

    # ── Hardware ──────────────────────────────────────────────────────────
    hardware = [_clean_item(l) for l in sections.get("hardware", [])]

    # ── Materials (clean + enrich) ────────────────────────────────────────
    materials_lines = [_clean_item(l) for l in sections.get("materials", [])]
    try:
        materials_enriched = enrich_materials(materials_lines)
    except Exception as exc:
        materials_enriched = [{"name": ln, "notes": f"enrich failed: {exc}"} for ln in materials_lines]

    # ── Procedure ────────────────────────────────────────────────────────
    procedure_paras = sections.get("procedure", [])
    proc_struct = gpt_steps(procedure_paras) if robot else []

    return {
        "hardware": hardware,
        "materials": materials_lines,
        "materials_enriched": materials_enriched,
        "procedure": procedure_paras,
        "procedure_structured": proc_struct,
        "raw_answer": answer_text.strip()
    }

# ════════════════════  5. CLI convenience  ════════════════════════════════ #
if __name__ == "__main__":
    import argparse, pathlib, sys, pprint
    ap = argparse.ArgumentParser(description="Convert answer text to JSON.")
    ap.add_argument("file", help="txt file containing NanoChemGPT answer")
    ap.add_argument("--robot", action="store_true", help="call OpenAI to split into atomic steps")
    ns = ap.parse_args()

    txt = pathlib.Path(ns.file).read_text(encoding="utf-8", errors="ignore")
    try:
        out = convert_to_json(txt, robot=ns.robot)
        json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    except ParserError as e:
        print("ParserError:", e, file=sys.stderr)
        sys.exit(1)
