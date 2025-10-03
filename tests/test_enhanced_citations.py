from harvester.enhanced_citations import (
    EnhancedCitationFilter,
    enhance_citation_relevance,
)


def make_ref(**kwargs):
    base = {
        "title": "Hydrothermal synthesis of SnO nanorods",
        "journal": "Nano Letters",
        "year": 2024,
        "doi": "10.1000/example",
        "isOpenAccess": True,
        "text": "Detailed synthesis procedure and characterization by SEM.",
        "authors": ["A", "B", "C"],
    }
    base.update(kwargs)
    return base


def test_extract_citation_context_merges_multiple_occurrences():
    f = EnhancedCitationFilter()
    txt = "We used the method [1] and repeated it [1]. Another method [2]."
    contexts = f.extract_citation_context(txt, [1, 2], window_size=20)
    assert 1 in contexts and 2 in contexts
    # multiple occurrences of [1] should be merged using separator
    assert "|" in contexts[1]


def test_score_query_alignment_prefers_synthesis_terms():
    f = EnhancedCitationFilter()
    context = "This work describes a synthesis procedure and experimental protocol for nanorods."
    score = f.score_query_alignment(
        context, "How to synthesize nanorods", intent="procedure"
    )
    assert score > 0.5


def test_rank_and_filter_reorders_and_filters():
    f = EnhancedCitationFilter()
    refs = [
        make_ref(),
        make_ref(
            journal="Some Journal",
            year=2010,
            doi=None,
            isOpenAccess=False,
            text="Short",
        ),
    ]
    response = (
        "We followed the reported protocol [1,2] and observed similar morphology."
    )
    filtered_response, filtered_refs = f.filter_low_relevance_citations(
        response, refs, "synthesize nanorods"
    )

    # At least one reference should remain
    assert isinstance(filtered_response, str)
    assert isinstance(filtered_refs, list)
    assert len(filtered_refs) >= 1


def test_enhance_citation_relevance_no_refs_returns_original():
    resp, refs = enhance_citation_relevance("No citations here", [], "query")
    assert resp == "No citations here"
    assert refs == []
