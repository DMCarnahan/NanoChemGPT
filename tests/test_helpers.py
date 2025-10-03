import pytest

from app_utils.pdf_utils import normalize_pdf_text
from app_utils.text_chunks import pick_method_paragraph, best_chunks_from_text
from app_utils.fact_extractor import extract_facts_from_text
from app_utils.rendering import render_protocol_md


def test_normalize_pdf_text_basic():
    s = "FeSO4\x03 7H2O -\n 100 mL"
    out = normalize_pdf_text(s)
    assert "FeSO4" in out
    assert "100 mL" in out


def test_pick_method_paragraph():
    text = "Intro\n\nHeat to 100 °C in a water bath. Stir.\n\nResults"
    p = pick_method_paragraph(text)
    assert "water bath" in p.lower()


def test_best_chunks_from_text():
    text = "Step 1: Mix.\nStep 2: Heat to 50 °C in a water bath.\nStep 3: Cool."
    chunks = best_chunks_from_text(text, "heat water")
    assert len(chunks) >= 1
    assert any("water bath" in c.lower() for c in chunks)


def test_extract_and_render_facts():
    t = "Prepare 100 mL of 0.1 M FeSO4 in a water bath at 50 °C. Dry at 60 °C for 2 h."
    facts = extract_facts_from_text(t)
    assert isinstance(facts, dict)
    md = render_protocol_md(facts)
    assert "Materials" in md or "Procedure" in md
