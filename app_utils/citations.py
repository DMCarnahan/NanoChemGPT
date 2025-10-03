from __future__ import annotations

from typing import List

from ref_utils import (
    DEFAULT_NANOCHEM_TERMS,
    dedupe_and_rerank,
    extract_used_ref_indexes,
    split_used_refs,
)


def build_references_payload(
    answer_text: str, refs_input: List[dict], *, question: str = "", top_k: int = 40
) -> dict:
    try:
        refs_all = dedupe_and_rerank(
            question or "",
            refs_input or [],
            domain_terms=DEFAULT_NANOCHEM_TERMS,
            top_k=max(top_k, len(refs_input or [])),
        )
    except Exception:
        refs_all = list(refs_input or [])

    try:
        used = extract_used_ref_indexes(answer_text or "")
    except Exception:
        used = []

    try:
        refs_used, index_map = split_used_refs(refs_all, used)
    except Exception:
        refs_used, index_map = list(refs_all), {
            i + 1: i + 1 for i in range(len(refs_all))
        }

    return {
        "refs_all": refs_all,
        "refs_used": refs_used,
        "index_map": index_map,
        "candidates": refs_all,
    }
