from app_utils.citations import build_references_payload
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
