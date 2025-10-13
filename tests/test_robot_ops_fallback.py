import json
import types
import pytest

from converter import convert_text_to_robot_ops

# We will simulate the app logic fallback by directly calling converter on a pseudo-protocol
# constructed similarly to the fallback path in app.py.

RAW_UNSTRUCTURED = """The iron oxide (magnetite) catalyst was synthesized via controlled precipitation. A 0.1 M FeSO4·7H2O solution (100 mL) was prepared and held at 45 C with stirring. 0.45 M NaOH was added at 5 mL/min until pH 12. The precipitate was filtered and dried at 50 C for 24 h."""


def _build_pseudo(text: str) -> str:
    # mimic fallback numbering (each sentence becomes a step)
    import re

    frags = [s.strip() for s in re.split(r"[\.\n]+", text) if s.strip()]
    numbered = []
    for i, s in enumerate(frags, start=1):
        if not s.endswith("."):
            s += "."
        numbered.append(f"{i}. {s}")
    return "1. **Procedure**:\n" + "\n".join(numbered)


def test_robot_ops_fallback_basic():
    pseudo = _build_pseudo(RAW_UNSTRUCTURED)
    doc = convert_text_to_robot_ops(pseudo)
    assert isinstance(doc, dict)
    assert "steps" in doc and doc["steps"], "Expected steps extracted in fallback"
    # Expect at least an add / stir / postprocess or heat element captured
    actions = {(s.get("action") or "").lower() for s in doc["steps"]}
    assert any(
        a in actions for a in ("stir", "heat_hold", "postprocess", "add", "add_solvent")
    ), actions
    # Verify micro_plan created
    assert (
        isinstance(doc.get("micro_plan"), list) and doc["micro_plan"]
    ), "micro_plan should not be empty"
    # Ensure defaults inserted
    d = doc.get("defaults") or {}
    assert d.get("stir_rpm") and d.get(
        "centrifuge_rpm"
    ), "Defaults missing required rpm values"


def test_robot_ops_fallback_contains_temperature_or_ph():
    pseudo = _build_pseudo(RAW_UNSTRUCTURED)
    doc = convert_text_to_robot_ops(pseudo)
    temps = [s.get("temperature_C") for s in doc["steps"] if s.get("temperature_C")]
    # We might parse at least one temperature (45 C or 50 C)
    assert any(t in (45, 50) for t in temps), temps
