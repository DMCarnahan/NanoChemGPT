from app_utils.citations import (
    build_references_payload,
    citation_repair_target,
    grounded_reference_indexes,
    needs_citation_repair,
)
from app_utils.jobs import get_job, mark_done, set_job
from app_utils.utils import doc, extract_used_markers, s, stringify_keys, wants_verbatim


def test_jobs_roundtrip():
    jid = "t1"
    set_job(jid, status="processing", progress=10)
    j = get_job(jid)
    assert j["status"] == "processing"
    mark_done(jid)
    assert get_job(jid)["status"] == "done"


def test_utils_s_and_doc():
    assert s(None) == ""
    assert isinstance(stringify_keys({"a": 1}), dict)
    assert isinstance(doc({"_id": 123, "ts": None}), dict)


def test_extract_used_markers_and_verbatim():
    res = extract_used_markers("This cites [1] and [2]. [CTX]")
    assert "refs" in res
    assert res["tags"]["CTX"] >= 1
    assert wants_verbatim("please transcribe verbatim") is True


def test_citations_build():
    payload = build_references_payload("No refs here", [])
    assert isinstance(payload, dict)


def test_citation_indexes_follow_reranked_reference_order():
    references = [
        {"title": "Unrelated clinical report", "index": 1},
        {"title": "Gold nanoparticle synthesis", "index": 2},
    ]

    payload = build_references_payload(
        "Use the reported synthesis [1].",
        references,
        question="gold nanoparticle synthesis",
    )

    assert [reference["index"] for reference in payload["refs_all"]] == [1, 2]
    assert payload["refs_used"][0]["title"] == "Gold nanoparticle synthesis"


def test_citation_repair_targets_two_grounded_sources_when_available():
    context = """<<<CTX_WEB>>>
[1] First source
Evidence one.

[2] Second source
Evidence two.

[A1.1] attachment
Attachment evidence.
"""
    grounded = grounded_reference_indexes(context, maximum=4)

    assert grounded == [1, 2]
    assert citation_repair_target(grounded) == 2
    assert needs_citation_repair([1], grounded) is True
    assert needs_citation_repair([1, 2], grounded) is False


def test_citation_repair_does_not_force_a_second_unavailable_source():
    grounded = grounded_reference_indexes("[3] Only grounded source\nEvidence.")

    assert citation_repair_target(grounded) == 1
    assert needs_citation_repair([3], grounded) is False
