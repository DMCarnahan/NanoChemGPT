"""
converter.py  –  Turn a formatted “NanoChemGPT” answer into explicit JSON
-----------------------------------------------------------------------

• Normalises “Hardware & Glassware”, “Hardware”, or “Glassware” → `sections["hardware"]`
• Strips leading bullets and trailing `[CTX] / [GEN] / [n]` tags from materials & hardware.
• Uses modern OpenAI function‑calling (no priming hack, parses messages with `role="tool"`).
"""

from __future__ import annotations
import os, re, json, textwrap
from typing import List, Dict, Any

import openai

# ════════════════════ 0. OpenAI client ════════════════════════════════════ #
_openai_api_key = os.getenv("OPENAI_API_KEY", "")
_client = openai.OpenAI(api_key=_openai_api_key) if _openai_api_key else None

# ════════════════════ 1. Regex helpers ════════════════════════════════════ #
_SUB_RX   = re.compile(r"^\s*\d+\.\s*\*\*(?P<name>[^*]+?)\*\*[:\s]*$", re.M)

_BULLET_RX = re.compile(r"^\s*[-*•–—]\s*")
_TAG_RX    = re.compile(r"""\s*\[(?:CTX|DB|PARSED|GEN|\d+)\]\s*[.。;:,-]?\s*$""")

def _clean_item(line: str) -> str:
    """Remove leading bullets and trailing citation / provenance tags."""
    return _TAG_RX.sub("", _BULLET_RX.sub("", line)).strip()

def _split_sections(txt: str) -> Dict[str, List[str]]:
    """Return a dict with keys hardware / materials / procedure / other."""
    sections: Dict[str, List[str]] = {k: [] for k in
                                      ("hardware", "materials", "procedure", "other")}
    current = "other"

    for raw in txt.splitlines():
        m = _SUB_RX.match(raw)
        if m:
            name = m.group("name").lower().strip()
            if name.startswith("hardware") or name == "glassware":
                current = "hardware"
            elif name.startswith("material"):
                current = "materials"
            elif name.startswith("procedure"):
                current = "procedure"
            else:
                current = "other"
            continue
        sections[current].append(raw.rstrip())

    for k in sections:
        sections[k] = [l for l in sections[k] if l.strip()]
    return sections


# ════════════════════ 2. Material enrichment (best‑effort) ════════════════ #
def enrich_materials(lines: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for ln in lines:
        if " as " in ln:
            name, notes = ln.split(" as ", 1)
            out.append({"name": name.strip(),
                        "notes": _clean_item(notes)})
        else:
            out.append({"name": ln})
    return out


# ════════════════════ 3. Function‑calling for atomic steps ════════════════ #
_FN_SCHEMA = {
    "name": "add_step",
    "description": "Add an explicit atomic step to the procedure",
    "parameters": {
        "type": "object",
        "properties": {
            "action":   {"type": "string"},
            "details":  {"type": "string"},
            "reagents": {"type": "array", "items": {"type": "string"}},
            "solvents": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["action", "details"]
    }
}

_SYSTEM_STEPS = textwrap.dedent("""    You are a chemistry protocol parser.
    Break the PROCEDURE text into explicit atomic steps,
    calling the add_step tool once for each distinct action.
""").strip()

def gpt_steps(paragraphs: List[str], model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    if not _client or not paragraphs:
        return []

    messages = [
        {"role": "system", "content": _SYSTEM_STEPS},
        {"role": "user",   "content": "PROCEDURE:\n" + "\n".join(paragraphs)}
    ]

    resp = _client.chat.completions.create(
        model=model,
        temperature=0,
        tools=[{"type": "function", "function": _FN_SCHEMA}],
        messages=messages,
    )

    steps: List[Dict[str, Any]] = []
    for ch in resp.choices:
        if ch.message.role == "tool":
            steps.append(json.loads(ch.message.content))
        elif ch.message.tool_calls:  # fallback for older models
            for tc in ch.message.tool_calls:
                steps.append(json.loads(tc.function.arguments))
    return steps


# ════════════════════ 4. Public API ═══════════════════════════════════════ #
class ParserError(RuntimeError):
    """Raised when answer cannot be parsed into expected sections."""


def convert_to_json(answer_text: str, *, robot: bool = False) -> Dict[str, Any]:
    secs = _split_sections(answer_text)

    hardware  = [_clean_item(l) for l in secs["hardware"]]
    materials = [_clean_item(l) for l in secs["materials"]]
    try:
        materials_enriched = enrich_materials(materials)
    except Exception as exc:  # never fail the whole parse
        materials_enriched = [{"name": ln, "notes": f"enrich failed: {exc}"}
                              for ln in materials]

    procedure   = secs["procedure"]
    proc_struct = gpt_steps(procedure) if robot else []

    return {
        "hardware": hardware,
        "materials": materials,
        "materials_enriched": materials_enriched,
        "procedure": procedure,
        "procedure_structured": proc_struct,
        "raw_answer": answer_text.strip(),
    }


# ════════════════════ 5. CLI helper ═══════════════════════════════════════ #
if __name__ == "__main__":
    import argparse, pathlib, sys
    p = argparse.ArgumentParser(description="Convert answer text to JSON.")
    p.add_argument("file", help="answer.txt produced by NanoChemGPT")
    p.add_argument("--robot", action="store_true",
                   help="include OpenAI function‑calling split into atomic steps")
    ns = p.parse_args()

    txt = pathlib.Path(ns.file).read_text(encoding="utf‑8", errors="ignore")
    try:
        json.dump(convert_to_json(txt, robot=ns.robot),
                  sys.stdout, indent=2, ensure_ascii=False)
    except ParserError as exc:
        print("ParserError:", exc, file=sys.stderr)
        sys.exit(1)

# ───────────── 6. simple post-parser for ‘details’ ────────────── #
_TEMP_RX = re.compile(r"(?P<val>[-+]?\d+(?:\.\d+)?)\s*°?\s*C", re.I)
_TIME_RX = re.compile(
    r"(?P<val>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>h(?:ours?)?|min(?:utes?)?)", re.I)

def _mk_vessel_map(hardware: List[str]) -> Dict[str, str]:
    """{'250 mL beaker': 'V1', 'round-bottom flask': 'V2', …}"""
    vmap = {}
    for idx, hw in enumerate(hardware, 1):
        vmap[hw.lower()] = f"V{idx}"
    return vmap

def _extract_fields(step: Dict[str, Any], vmap: Dict[str, str]) -> Dict[str, Any]:
    if "details" not in step:
        return step
    txt = step["details"]

    # 1) temperature
    if "temperature" not in step:
        m = _TEMP_RX.search(txt)
        if m:
            step["temperature"] = m.group("val").strip()

    # 2) time  (rename duration to time for clarity)
    if "time" not in step and "duration" not in step:
        m = _TIME_RX.search(txt)
        if m:
            tval = m.group("val").strip()
            unit = m.group("unit").lower()
            step["time"] = f"{tval} {unit}"

    # 3) vessel
    if "vessel" not in step:
        for phrase, vid in vmap.items():
            if phrase in txt.lower():
                step["vessel"] = vid
                break

    # 4) clean up
    step.pop("duration", None)          # prefer 'time'
    step.pop("details",  None)          # drop the free text
    return step

# Hook it into convert_to_json
_orig_convert = convert_to_json  # keep reference

def convert_to_json(answer_text: str, *, robot: bool = False) -> Dict[str, Any]:  # type: ignore[override]
    data = _orig_convert(answer_text, robot=robot)

    vmap = _mk_vessel_map(data["hardware"])
    data["procedure_structured"] = [
        _extract_fields(step, vmap) for step in data["procedure_structured"]
    ]
    return data
