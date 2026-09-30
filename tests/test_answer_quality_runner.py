from ai_eval.answer_quality_runner import grade_answer


def test_grade_answer_accepts_valid_protocol():
    result = grade_answer(
        "## Synthesis Protocol:\nUse 25 mL in air [1].",
        {
            "mode": "protocol",
            "required_patterns": [r"25\s*mL", "in air"],
            "forbidden_patterns": ["argon"],
            "allowed_numeric_citations": [1],
        },
    )
    assert result == {"passed": True, "failures": [], "cited": [1]}


def test_grade_answer_reports_format_hallucination_and_bad_citation():
    result = grade_answer(
        "A definite 99% yield [2].\n## References\nMade up.",
        {
            "mode": "reasoning",
            "required_patterns": ["Evidence"],
            "forbidden_patterns": ["99%"],
            "allowed_numeric_citations": [1],
        },
    )
    assert result["passed"] is False
    assert "missing reasoning heading" in result["failures"]
    assert "answer included a References section" in result["failures"]
    assert "invented numeric citations: [2]" in result["failures"]
