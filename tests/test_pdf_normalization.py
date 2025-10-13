import pytest
from app_utils.pdf_utils import normalize_pdf_text


def test_collapsed_dotted_letters():
    noisy = "m·a·t·e·r·i·a·l p·r·o·p·e·r·t·i·e·s"
    assert normalize_pdf_text(noisy) == "material properties"


def test_preserve_hydrate():
    txt = "The sample was CuSO4·5H2O at room temperature."
    assert "CuSO4·5H2O" in normalize_pdf_text(txt)


def test_preserve_multiple_hydrates_and_strip_noise():
    noisy = "C·u·S·O·4· ·5·H·2·O mixed with Na2SO4·10H2O"
    out = normalize_pdf_text(noisy)
    assert "CuSO4·5H2O" in out
    assert "Na2SO4·10H2O" in out
    # Ensure no residual mid-dot between ordinary letters
    assert "C·u" not in out


def test_adduct_pattern():
    txt = "Formed a complex CoCl2·6NH3 under reflux."
    assert "CoCl2·6NH3" in normalize_pdf_text(txt)


def test_whitespace_and_hyphenation():
    txt = "multi-\n line with hyphen-\n ation"
    # 'multi-' line break treated as a hyphenated continuation -> join
    # 'hyphen-' + 'ation' likewise joined into 'hyphenation'
    assert normalize_pdf_text(txt) == "multiline with hyphenation"


def test_empty_input():
    assert normalize_pdf_text("") == ""
