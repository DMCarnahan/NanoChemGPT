from ref_utils import extract_used_ref_indexes, renumber_citations


def test_extract_used_ref_indexes_simple_and_ranges():
    txt = "Results were reported [1,2] and in range [3-5] and [7-9]."
    idxs = extract_used_ref_indexes(txt)
    # ensure ranges expanded for the later ranges
    assert 3 in idxs and 4 in idxs and 5 in idxs
    assert 7 in idxs and 8 in idxs and 9 in idxs
    assert all(isinstance(i, int) for i in idxs)


def test_renumber_citations_basic_mapping():
    text = "See methods [1] and review [3]."
    new = renumber_citations(text, {1: 5, 3: 7})
    assert "[5]" in new and "[7]" in new
