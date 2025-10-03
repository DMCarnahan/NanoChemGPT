from app_utils import utils as u


def test_s_returns_string():
    assert isinstance(u.s(None), str)
    assert u.s(123) == "123"


def test_safe_id_none_and_bad():
    assert u.safe_id(None) is None
    assert u.safe_id("") is None
    # random non-objectid string returns None (no exception)
    assert u.safe_id("notanid") is None


def test_stringify_keys_and_doc():
    obj = {"a": 1, "b": {"c": 2}}
    s = u.stringify_keys(obj)
    assert s["a"] == 1
    d = u.doc({"_id": 123, "ts": None})
    assert "_id" in d and isinstance(d["_id"], str) or d["_id"] is None


def test_extract_used_markers_and_wants_verbatim():
    txt = "See [1] and [CTX] markers and [PARSED] and [DB]"
    res = u.extract_used_markers(txt)
    assert "refs" in res and isinstance(res["refs"], list)
    assert u.wants_verbatim("please repeat verbatim") is True


def test_clean_verbatim_block_replaces_tokens():
    s = "[A1.1] attachment:file.pdf\nSome text" + ""
    out = u.clean_verbatim_block(s)
    assert "attachment" not in out


from app_utils import utils as u


def test_s_returns_string():
    assert isinstance(u.s(None), str)
    assert u.s(123) == "123"


def test_safe_id_none_and_bad():
    assert u.safe_id(None) is None
    assert u.safe_id("") is None
    # random non-objectid string returns None (no exception)
    assert u.safe_id("notanid") is None


def test_stringify_keys_and_doc():
    obj = {"a": 1, "b": {"c": 2}}
    s = u.stringify_keys(obj)
    assert s["a"] == 1
    d = u.doc({"_id": 123, "ts": None})
    assert "_id" in d and isinstance(d["_id"], str) or d["_id"] is None


def test_extract_used_markers_and_wants_verbatim():
    txt = "See [1] and [CTX] markers and [PARSED] and [DB]"
    res = u.extract_used_markers(txt)
    assert "refs" in res and isinstance(res["refs"], list)
    assert u.wants_verbatim("please repeat verbatim") is True


def test_clean_verbatim_block_replaces_tokens():
    s = "[A1.1] attachment:file.pdf\nSome text" + ""
    out = u.clean_verbatim_block(s)
    assert "attachment" not in out
