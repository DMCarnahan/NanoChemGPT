from __future__ import annotations
import json, re, textwrap, types, sys
from pathlib import Path
from typing import Dict, List, Any

__all__ = ["convert_to_json", "ParserError"]
SCHEMA_VERSION = "1.8.2"

# ---------------------------------------------------------------------------
# 0.  Guarantee the module imports even if on-device translator is absent
# ---------------------------------------------------------------------------
# Some P2A builds expect rxn_opennmt_py at import-time; stub it if missing.
sys.modules.setdefault("rxn_opennmt_py", types.ModuleType("rxn_opennmt_py"))

# ---------------------------------------------------------------------------
# 1.  Try to load the on-device translator. If unavailable, _P2A_MODEL=None.
# ---------------------------------------------------------------------------
try:
    from paragraph2actions.predictor import Paragraph2Actions

    _P2A_MODEL = Paragraph2Actions()          # heavy; loads once at import
except Exception:
    _P2A_MODEL = None                         # will fall back to cloud

# ---------------------------------------------------------------------------
# 2.  Cloud fallback (tiny helper). Only used when _P2A_MODEL is None
# ---------------------------------------------------------------------------
try:
    from p2a_translator import translate_paragraphs  # your HTTP helper
except ImportError:
    translate_paragraphs = None

# ---------------------------------------------------------------------------
# 3.  Regex helpers
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
# 4.  Exceptions
# ---------------------------------------------------------------------------
class ParserError(ValueError):
    """Raised when input text cannot be parsed or a dependency is missing."""

# ---------------------------------------------------------------------------
# 5.  Utility: map P2A action dict ➜ NanoChem step
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
# 6.  Core helper: paragraph text ➜ list of atomic steps
# ---------------------------------------------------------------------------
def _paragraphs_to_steps(paragraphs: list[str]) -> list[dict[str, Any]]:
    """Return structured steps using on-device model or hosted fallback."""
    if _P2A_MODEL:
        actions = _P2A_MODEL.predict("\n\n".join(paragraphs))
        return [_map_p2a_action_to_schema(a) for a in actions]

    if translate_paragraphs:
        actions = translate_paragraphs(paragraphs)
        return [_map_p2a_action_to_schema(a) for a in actions]

    raise ParserError(
        "Paragraph2Actions translator unavailable. "
        "Install rxn-opennmt-py or configure the cloud helper."
    )

# ---------------------------------------------------------------------------
# 7.  Public API: raw text ➜ NanoChem JSON
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

    # ---- 7.1  Split into named sections -----------------------------------
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

    # ---- 7.2  Atomic steps (optional) --------------------------------------
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

    # ---- 7.3  Compose output ----------------------------------------------
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
