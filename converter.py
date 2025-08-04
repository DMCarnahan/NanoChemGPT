from __future__ import annotations
import json, re, textwrap, types, sys, os, httpx
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI
from chem_post import postprocess_steps

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
            "action":      {"type": "string"},
            "details":     {"type": "string"},
            "reagents":    {"type": "array", "items": {"type": "string"}},
            "solvents":    {"type": "array", "items": {"type": "string"}},
            "temperature": {"type": "string"},
            "duration":    {"type": "string"},
            "rpm":         {"type": "string"},
            "atmosphere":  {"type": "string"},
        },
        "required": ["action", "details"],
    },
}

_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(trust_env=False, timeout=30.0),
)


def gpt_steps(paragraphs: list[str], model: str = "gpt-4o-mini") -> list[dict]:
    """
    Return a list of atomic-step dictionaries extracted via
    GPT-4o function-calling, one tool call per input line.
    """
    msgs = [{
        "role": "system",
        "content": (
            "You are a chemistry assistant. "
            "For every input line you receive, call the function "
            "`add_step` exactly once, filling its JSON arguments. "
            "If a line has multiple operations, split them into separate steps, "
            "calling `add_step` multiple times."
        ),
    }]

    # single “priming” example so the model sees the schema once
    msgs.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "name": "add_step",
            "arguments": json.dumps({
                "action": "add",
                "details": "Add 2 g KOH.",
                "reagents": ["KOH"],
            }),
        }],
    }) # type: ignore

    # one user message per numbered line
    for p in paragraphs:
        clean = re.sub(r"^\s*\d+[.)]\s*", "", p).strip()
        if clean:
            msgs.append({"role": "user", "content": clean})
print("[gpt_steps] paragraphs:", paragraphs)
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        tools=[{"type":"function","function":_fn_schema}],
        # force the model to always call the function:
        tool_choice={"type":"function","function":{"name":"add_step"}},
        messages=msgs,
    )

    steps = []
    for choice in resp.choices:
        if choice.finish_reason == "tool_calls":
            for tc in choice.message.tool_calls:
                steps.append(json.loads(tc.function.arguments))

# ---------------------------------------------------------------------------
# 5.  Public API: raw text ➜ NanoChem JSON
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

    # ---- 5.1  Split into named sections -----------------------------------
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

    # ---- 5.2  Atomic steps --------------------------------------
    procedure_structured: list[dict[str, Any]] = []
    if robot:
        try:
            procedure_structured = postprocess_steps(procedure_structured)
        except Exception as exc:
            procedure_structured = [{"action":"error","details":f"post-proc: {exc}"}]

    # ---- 5.3  Compose output ----------------------------------------------
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
