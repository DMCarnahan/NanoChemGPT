from __future__ import annotations

import re

from ref_utils import (
    DEFAULT_NANOCHEM_TERMS,
    dedupe_and_rerank,
    extract_used_ref_indexes,
    split_used_refs,
)

_EVIDENCE_SOURCE_HEADER = re.compile(r"(?m)^\[(\d+)\]\s+\S")
_CITATION_MARKER = re.compile(r"\[(?:\d+(?:\s*[-–,]\s*\d+)*|[AU]\d+(?:\.\d+)*)\]")


def preserves_non_citation_text(draft: str, repaired: str) -> bool:
    """A citation-only pass must not rewrite chemistry or execution conditions."""

    def wording(text):
        text = re.sub(r"\s+", " ", _CITATION_MARKER.sub("", text or "")).strip()
        return re.sub(r"\s+([.,;:!?])", r"\1", text)

    return wording(draft) == wording(repaired)


def _positive_indexes(values) -> set[int]:
    indexes = set()
    for value in values or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index > 0:
            indexes.add(index)
    return indexes


def grounded_reference_indexes(
    context: str, *, maximum: int | None = None
) -> list[int]:
    """Return distinct numbered literature sources that have actual evidence text."""

    indexes = {int(match) for match in _EVIDENCE_SOURCE_HEADER.findall(context or "")}
    if maximum is not None:
        indexes = {index for index in indexes if 1 <= index <= maximum}
    return sorted(indexes)


def citation_repair_target(
    grounded_indexes: list[int], *, maximum_sources: int = 2
) -> int:
    """Choose a bounded citation target without forcing unavailable sources."""

    available = len(_positive_indexes(grounded_indexes))
    return min(max(0, maximum_sources), available)


def needs_citation_repair(
    used_indexes: list[int], grounded_indexes: list[int], *, maximum_sources: int = 2
) -> bool:
    """Return whether fewer grounded sources are cited than the evidence permits."""

    grounded = _positive_indexes(grounded_indexes)
    cited_grounded = _positive_indexes(used_indexes) & grounded
    return len(cited_grounded) < citation_repair_target(
        list(grounded), maximum_sources=maximum_sources
    )


def build_references_payload(
    answer_text: str, refs_input: list[dict], *, question: str = "", top_k: int = 40
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

    refs_all = [
        {**reference, "index": index} for index, reference in enumerate(refs_all, 1)
    ]

    try:
        used = extract_used_ref_indexes(answer_text or "")
    except Exception:
        used = []

    try:
        refs_used, index_map = split_used_refs(refs_all, used)
    except Exception:
        refs_used, index_map = (
            list(refs_all),
            {i + 1: i + 1 for i in range(len(refs_all))},
        )

    return {
        "refs_all": refs_all,
        "refs_used": refs_used,
        "index_map": index_map,
        "candidates": refs_all,
    }
