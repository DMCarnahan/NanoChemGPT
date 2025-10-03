from harvester.enhanced_relevance import (
    EnhancedRelevanceFilter,
    enhance_harvester_relevance,
)


def make_paper(**kwargs):
    base = {
        "title": "Synthesis of SnO nanorods with controlled dimensions",
        "abstract": "We report synthesis and characterization of SnO nanorods.",
        "text": "The synthesis used hydrothermal methods. Characterization by SEM and XRD confirmed morphology.",
        "journal": "Nano Letters",
        "year": 2024,
        "doi": "10.1234/example.doi",
        "isOpenAccess": True,
        "entities": [
            {"label": "MATERIAL", "text": "SnO"},
            {"label": "ACTION", "text": "synthesis"},
            {"label": "TEMP", "text": "180 C"},
        ],
        "keywords": ["SnO", "nanorod", "hydrothermal"],
    }
    base.update(kwargs)
    return base


def test_score_components_are_reasonable():
    f = EnhancedRelevanceFilter(min_year=2018, quality_threshold=0.0)
    paper = make_paper()

    score = f.calculate_relevance_score(paper, query_terms={"sno", "sn"})

    # component scores should be floats between 0 and 1
    assert 0.0 <= score.content_score <= 1.0
    assert 0.0 <= score.domain_score <= 1.0
    assert 0.0 <= score.recency_score <= 1.0
    assert 0.0 <= score.entity_score <= 1.0
    assert 0.0 <= score.quality_score <= 1.0
    assert 0.0 <= score.total_score <= 1.0


def test_filter_papers_filters_by_threshold():
    f = EnhancedRelevanceFilter(min_year=2018, quality_threshold=0.5)
    good = make_paper()
    bad = make_paper(
        title="Unrelated review",
        text="This is about ecology.",
        journal="PLOS One",
        year=2017,
        doi=None,
        isOpenAccess=False,
        entities=[],
    )

    results = f.filter_papers([good, bad], query_terms={"sno"})
    # Only the good paper should pass the 0.5 threshold
    assert len(results) == 1
    assert results[0][0]["title"] == good["title"]


def test_enhance_harvester_relevance_adds_metadata():
    papers = [make_paper()]
    enhanced = enhance_harvester_relevance(
        papers, config={"min_year": 2018, "quality_threshold": 0.0}
    )

    assert isinstance(enhanced, list)
    assert "relevance_score" in enhanced[0]
    assert "relevance_breakdown" in enhanced[0]
    assert "relevance_reasons" in enhanced[0]
