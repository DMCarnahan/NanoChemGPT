from __future__ import annotations
import json, re, textwrap, types, sys
from pathlib import Path
from typing import Dict, List, Any

__all__ = ["convert_to_json", "ParserError"]
SCHEMA_VERSION = "1.8.2"

# ---------------------------------------------------------------------------
# 1.  Load the on-device translator. 
# ---------------------------------------------------------------------------
from chemactor import ActionExtractor
_ACTOR = ActionExtractor("en_core_actions")   # loads weights from cache
                         # will fall back to cloud

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
# 4.  Utility: map chemactor action dict ➜ NanoChem step
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
def _paragraphs_to_steps(paragraphs: list[str]) -> list[dict]:
    """Return NanoChem step dicts for each paragraph."""
    steps = []
    for para in paragraphs:
        for act in _ACTOR(para):
            step = {
                "action": act.operation.lower(),
                "details": act.text,
            }
            # map ChemActor fields → schema
            if act.reagents:
                step["reagents"] = act.reagents
            if act.solvents:
                step["solvents"] = act.solvents
            if act.temperature:
                step["temperature"] = act.temperature
            if act.duration:
                step["duration"] = act.duration
            steps.append(step)
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
            procedure_structured = _paragraphs_to_steps(paragraphs)
        except Exception as exc:
            procedure_structured = [
                {"action": "error", "details": str(exc)}
            ]
    else:
        # simple fallback: each line as a step
        procedure_structured = [{"action": ln} for ln in sections.get("procedure", [])]

    # ---- 6.3  Compose output ----------------------------------------------
    return {
        "schema_version": SCHEMA_VERSION,
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
