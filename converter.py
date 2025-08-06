from __future__ import annotations
import json, re, textwrap, types, sys, os, httpx
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI
from chem_post import postprocess_steps
from chem_tools import enrich_materials

# ---------------------------------------------------------------------------
# 1.  Regex helpers
# ---------------------------------------------------------------------------
_HEADING_LINE = re.compile(
    r"""^\s*
        (?:\d+[\.\)]\s*)?
        (?:\*\*)?
        (?P<name>hardware(?:\s*&\s*glassware)?|glassware|
                  materials|reagents|procedure|steps|method|title)
        (?:\*\*)?
        \s*:?\s*$""",
    re.I | re.VERBOSE,
)
_LIST_BULLET = re.compile(r"^\s*(?:[-*•–—]\s+|\d+[\.\)]\s+)")

# ---------------------------------------------------------------------------
# 2.  Exceptions
# ---------------------------------------------------------------------------
class ParserError(ValueError):
    """Raised when input text cannot be parsed or a dependency is missing."""

# ---------------------------------------------------------------------------
# 3.  Utility: map action dict ➜ NanoChem step
# ---------------------------------------------------------------------------
def _map_p2a_action_to_schema(act: dict[str, Any]) -> dict[str, Any]:
    op = (act.get("operation") or act.get("action") or "action").lower()
    step: dict[str, Any] = {"action": op, "details": act.get("text", "")}

    for key in (
        "reagents",
        "solvents",
        "vessel",
        "temperature",
        "duration",
        "dropwise",
        "atmosphere",
        "pressure",
        "rpm",
        "rate",
        "ph",
        "yield",
    ):
        if key in act and act[key]:
            step[key] = act[key]

    # remove empties
    return {k: v for k, v in step.items() if v not in ("", None, [], {})}

# ---------------------------------------------------------------------------
# 4.  Core helper: paragraph text ➜ list of atomic steps
# ---------------------------------------------------------------------------
_fn_schema = {
    "name": "add_step",
    "description": "Add ONE atomic operation in a chemical synthesis step.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "details": {"type": "string"},
            "reagents": {"type": "array","items":{"type":"string"}},
            "solvents": {"type": "array","items":{"type":"string"}},
            "temperature": {"type": "string"},
            "duration": {"type": "string"},
            "rpm": {"type": "string"},
            "atmosphere": {"type": "string"},
        },
        "required": ["action","details"]
    },
}

_client = OpenAI(
api_key=os.getenv("OPENAI_API_KEY"),
http_client=httpx.Client(trust_env=False, timeout=30.0),
)

def gpt_steps(paragraphs, model="gpt-4o-mini"):
    msgs = [{
        "role": "system",
        "content": (
            "You are a chemistry assistant. For every input line you receive, "
            "call the function `add_step` exactly once, filling its JSON arguments. "
            "If a line has multiple operations, split them into separate steps, "
            "calling `add_step` multiple times."
        ),
    }]

    msgs.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_priming",
            "type": "function",
            "function": {
                "name": "add_step",
                "arguments": json.dumps({
                    "action": "add",
                    "details": "Add 2 g KOH.",
                    "reagents": ["KOH"]
                })
            }
        }]
    }) # type: ignore

    # One user message per step line
    for p in paragraphs:
        clean = re.sub(r"^\s*\d+[.)]\s*", "", p).strip()
        if clean:
            msgs.append({"role": "user", "content": clean})

    resp = _client.chat.completions.create(
        model=model,
        temperature=0,
        tools=[{"type":"function","function":_fn_schema}],
        tool_choice={"type":"function","function":{"name":"add_step"}},  # force it
        messages=msgs,
    )

    steps = []
    for ch in resp.choices:
        if getattr(ch, "finish_reason", None) == "tool_calls" and ch.message.tool_calls:
            for tc in ch.message.tool_calls:
                steps.append(json.loads(tc.function.arguments))
    return steps

# ---------------------------------------------------------------------------
# 5.  Public API: raw text ➜ NanoChem JSON
# ---------------------------------------------------------------------------
def convert_to_json(raw: str, *, robot: bool = False) -> Dict[str, Any]:
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")
    raw = textwrap.dedent(raw).strip()

    # ---- Split into named sections ----
    sections: Dict[str, List[str]] = {}
    current = None
    for line in raw.splitlines():
        m = _HEADING_LINE.match(line)
        if m:
            current = m.group("name").lower()
            sections[current] = []
            continue
        if current and (_LIST_BULLET.match(line) or line.strip()):
            sections[current].append(line.strip())

    # ---- Atomic steps ----
    procedure_structured: List[Dict[str, Any]] = []
    if sections.get("procedure"):
        if robot:
            paragraphs = [ln for ln in sections["procedure"] if ln.strip()]
            print("[convert] calling gpt_steps on", len(paragraphs), "lines")
            try:
                steps = gpt_steps(paragraphs, model="gpt-4o-mini")
                procedure_structured = postprocess_steps(steps)  # one pass
            except Exception as exc:
                procedure_structured = [{"action": "error", "details": f"extract: {exc}"}]
        else:
            # minimal fallback: one step per line
            procedure_structured = [{"action": "step", "details": ln}
                                    for ln in sections["procedure"]]

    # ---- Materials enrichment (best-effort) ----
    materials_lines = sections.get("materials", [])
    try:
        materials_struct = enrich_materials(materials_lines)
    except Exception as exc:
        materials_struct = [{"name": ln, "notes": f"enrich failed: {exc}"} for ln in materials_lines]

    # ---- Title ----
    title = sections.get("title", ["SynthesisProtocol"])
    title = title[0] if title else "SynthesisProtocol"

    # ---- Compose output ----
    return {
        "title": title,
        "hardware": sections.get("hardware", []),
        "materials": materials_lines,
        "materials_enriched": materials_struct,
        "procedure": sections.get("procedure", []),
        "procedure_structured": procedure_structured,
    }

# Stand-alone CLI use ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python converter.py <protocol.txt> [robot]")
    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    print(
        json.dumps(
            convert_to_json(text, robot=len(sys.argv) > 2),
            indent=2,
            ensure_ascii=False,
        )
    )
