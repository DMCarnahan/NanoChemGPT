from __future__ import annotations
import json, re, textwrap, types, sys, os, httpx
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI
# ---------------------------------------------------------------------------
# 1.  OpenAI Call 
# ---------------------------------------------------------------------------

_ACTION_FN = {
    "name": "add_step",
    "description": "Add an atomic operation in a chemical synthesis procedure.",
    "parameters": {
        "type": "object",
        "properties": {
            "action":   {"type": "string", "description": "verb / operation"},
            "details":  {"type": "string", "description": "full sentence"},
            "reagents": {"type": "array",  "items":{"type":"string"}},
            "solvents": {"type": "array",  "items":{"type":"string"}},
            "temperature":{"type":"string"},
            "duration": {"type":"string"},
            "rpm":      {"type":"string"},
            "atmosphere":{"type":"string"},
        },
        "required": ["action", "details"],
    },
}

# ---------------------------------------------------------------------------
# 2.  Regex helpers
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
# 3.  Exceptions
# ---------------------------------------------------------------------------
class ParserError(ValueError):
    """Raised when input text cannot be parsed or a dependency is missing."""

# ---------------------------------------------------------------------------
# 4.  Utility: map action dict ➜ NanoChem step
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
# 5.  Core helper: paragraph text ➜ list of atomic steps
# ---------------------------------------------------------------------------
_no_proxy_client = httpx.Client(trust_env=False, timeout=30.0)
oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                   http_client=_no_proxy_client)

def gpt_steps(paragraph: str, model:str="gpt-4o-mini") -> list[dict]:
    resp = oa_client.chat.completions.create(
        model=model,
        temperature=0,
        tools=[{"type":"function", "function":_ACTION_FN}],
        tool_choice={"type":"function", "function":{"name":"add_step"}},
        messages=[
            {"role":"system","content":"You extract structured synthesis steps."},
            {"role":"user","content":paragraph}
        ],
    )
    steps = []
    for choice in resp.choices:
        if choice.finish_reason == "tool_calls":
            payload = json.loads(choice.message.tool_calls[0].function.arguments)
            steps.append(payload)
    return steps

# ---------------------------------------------------------------------------
# 6.  Public API: raw text ➜ NanoChem JSON
# ---------------------------------------------------------------------------
def convert_to_json(raw: str, *, robot: bool = False) -> Dict[str, Any]:
    """
    Convert a free-form protocol into structured NanoChemGPT JSON.

    Parameters
    ----------
    raw : str
        Input protocol text.
    robot : bool, optional
        If True, attempt atomic step extraction.  Falls back gracefully if the
        translator is missing.

    Returns
    -------
    dict
        JSON compatible with NanoChemGPT UI.
    """
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")

    raw = textwrap.dedent(raw).strip()

    # ---- 6.1  Split into named sections -----------------------------------
    sections: dict[str, list[str]] = {}
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

    # ---- 6.2  Atomic steps --------------------------------------
    procedure_structured: list[dict[str, Any]] = []
    if robot and "procedure" in sections:
        paragraphs = textwrap.dedent("\n".join(sections["procedure"])).split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        try:
            procedure_structured = []
            for para in paragraphs:
                procedure_structured.extend(gpt_steps(para))
        except Exception as exc:
            procedure_structured = [{"action":"error","details":str(exc)}]

    # ---- 6.3  Compose output ----------------------------------------------
    return {
        "title": sections.get("title", ["SynthesisProtocol"])[0]
        if "title" in sections
        else "SynthesisProtocol",
        "hardware": sections.get("hardware", []),
        "materials": sections.get("materials", []),
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
