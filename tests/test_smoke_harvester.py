import sys
import types

# Provide a minimal fake html_text module for environments where it's not installed
_html_mod = types.ModuleType("html_text")


def _extract_text(html):
    return ""


_html_mod.extract_text = _extract_text
sys.modules.setdefault("html_text", _html_mod)

# The harvester package ships a local utils.py but imports it as 'utils'.
# Provide a shim mapping 'utils' -> harvester.utils to satisfy imports during tests.
import importlib

try:
    hu = importlib.import_module("harvester.utils")
    sys.modules.setdefault("utils", hu)
except Exception:
    pass


# Create lightweight stubs for other top-level modules that harvester imports
def _make_stub(name, attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)


_make_stub("arxiv_api", {"search_arxiv": lambda q, n: []})
_make_stub(
    "eupmc_api",
    {"search_eupmc": lambda q, y, n: [], "fetch_fulltext_jats": lambda pmc: None},
)
_make_stub("unpaywall_api", {"unpaywall_lookup": lambda doi: {}})
_make_stub("grobid_client", {"pdf_to_tei": lambda url, b: None})
_make_stub(
    "tei_utils",
    {"tei_to_sections": lambda t: [], "filter_methods_sections": lambda s: []},
)
_make_stub(
    "jats_utils",
    {"jats_to_sections": lambda j: [], "filter_methods_sections": lambda s: []},
)
_make_stub("oa_resolver", {"resolve_oa": lambda doi: {"is_oa": False}})
_make_stub(
    "enhanced_relevance", {"enhance_harvester_relevance": lambda papers, cfg: papers}
)


# Miner runtime: get_miner should return an object with extract_procedure
class _FakeMiner:
    def extract_procedure(self, txt):
        return {"operations": [], "expanded": []}


_make_stub("miner.runtime", {"get_miner": lambda nlp_model=None: _FakeMiner()})

from harvester.harvester import (
    _arxiv_id_from_any,
    _authors_to_list,
    _norm_doi,
    fallback_methods_from_text,
    score_paragraph,
    split_paragraphs,
)


def test_norm_doi_and_arxiv():
    doi = "https://doi.org/10.1234/ABC.DEF/01"
    assert _norm_doi(doi).startswith("10.1234/")

    arxiv = "https://arxiv.org/abs/2020.12345"
    assert _arxiv_id_from_any(arxiv) == "2020.12345"


def test_authors_and_paragraphs():
    authors = "Alice; Bob and Carol, Dave"
    lst = _authors_to_list(authors)
    assert isinstance(lst, list) and len(lst) >= 2

    text = "First para.\n\nSecond para with more text."
    paras = split_paragraphs(text)
    assert len(paras) == 2


def test_score_and_fallback_methods():
    proc = (
        "Add 5 mL of reagent, heat to 80 °C for 2 h and stir. "
        "Then cool to room temperature, centrifuge at 3000 rpm for 10 min, "
        "wash with ethanol, and dry under vacuum."
    )
    sc = score_paragraph(proc)
    assert sc > 0

    doc = """
    Methods
    We prepared samples by mixing reagents and annealing.

    Results
    Nothing more.
    """
    fm = fallback_methods_from_text(doc)
    assert fm is not None and "Methods" in fm
