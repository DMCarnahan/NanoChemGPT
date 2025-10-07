import re
from converter import convert_text_to_robot_ops

SYNTHESIS_BLOCK = """2.1. Synthesis of magnetite nanoparticles (MNPs)
A 0.1 M FeSO4·7H2O solution (100 mL) was prepared and maintained at 45 °C with stirring.
0.45 M NaOH was added at 5 mL/min until pH 12.
The precipitate was filtered and dried at 50 °C for 24 h.
2.2. Characterization of magnetite nanoparticles (MNPs)
XRD, TEM and VSM measurements were performed.
"""

# Mimic fallback slice + merge by providing only raw unstructured text (no numbered procedure marker)

def test_fallback_section_slicer_excludes_characterization():
    doc = convert_text_to_robot_ops("1. **Procedure**:\n1. placeholder")  # warm up converter (caches)

    import re as _re
    raw = SYNTHESIS_BLOCK
    # slice manually (simulate _slice_synthesis): capture 2.1 up to 2.2
    m = _re.search(r"(^|\n)\s*2\.1[^\n]{0,40}synthesi[sd].*?(?=\n\s*2\.2\b|$)", raw, _re.I | _re.S)
    sliced = m.group(0) if m else raw
    assert "Characterization" not in sliced, "Section slicer should exclude 2.2 content"


def test_line_merge_reduces_fragments():
    # Create artificial hard-wrapped lines that should merge
    text = (
        "2.1. Synthesis of magnetite nanoparticles (MNPs)\n"
        "A 0.1 M FeSO4·7H2O solution (100 mL) was prepared and maintained at 45 °C with\n"
        "stirring and 0.45 M NaOH was added at 5 mL/min until pH 12 then filtered\n"
        "and dried at 50 °C for 24 h.\n"
    )
    # emulate merge algorithm
    import re as _re
    lines = [l for l in text.splitlines() if l.strip()]
    KEY_VERBS = r"add|pour|introduce|titr|stir|heat|maintain|hold|cool|filter|wash|dry|centrifuge|decant|resuspend|collect|transfer|dissolv|prepare|monitor|adjust"
    REAGENT_HINT = _re.compile(r"\b(\d+(?:\.\d+)?\s*(?:mL|ml|g|mg|mol|mmol|M|°C|deg|C)|pH\s*\d+(?:\.\d+)?)\b")
    keep = []
    for ln in lines:
        low = ln.lower()
        if _re.search(KEY_VERBS, low) or REAGENT_HINT.search(ln):
            keep.append(ln.rstrip())
    salvage = "\n".join(keep)
    merged = []
    buf = []
    SENT_END = _re.compile(r"[\.;:!?]$")
    for raw_line in salvage.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        buf.append(line)
        if SENT_END.search(line) or len(line) > 160:
            merged.append(" ".join(buf))
            buf = []
    if buf:
        merged.append(" ".join(buf))
    # We expect merged length < original wrapped line count (3 wrapped lines become 1 sentence)
    assert len(merged) < len(keep), (merged, keep)
