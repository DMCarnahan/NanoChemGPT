"""Text chunking and method-paragraph selection helpers."""

from __future__ import annotations

import re
from typing import List

from .pdf_utils import normalize_pdf_text


def best_chunks_from_text(
    text: str, query: str, max_chunk_chars: int = 1200, top_k: int = 3
) -> List[str]:
    if not text:
        return []
    text = normalize_pdf_text(text)

    q_tokens = {t for t in re.findall(r"[A-Za-z0-9]{3,}", (query or "").lower())}

    cues = [
        r"\b0\.1\s*m\b",
        r"\b0\.45\s*m\b",
        r"\b\d+\s*m[lL]\s*/?\s*min\b",
        r"\bpH\s*(?:\d+(?:\.\d+)?)\b",
        r"\b\d+\s*°\s*C\b",
        r"\b\d+\s*h\b",
        r"\bfiltered?\b|\bfiltration\b|\bvacuum\b",
        r"\bdry(?:ing|ed)?\b",
        r"\bautotitrator\b|\bmettler\b|\bdl50\b",
        r"\bwater bath\b",
        r"\bfe\s*so\s*4\b|\bfeooh\b|\bmagnetite\b|\bfe3o4\b",
        r"\bprocedure\b|\bmethod\b|\bsynthesi",
    ]
    cue_res = [re.compile(c, re.I) for c in cues]

    paras = [p.strip() for p in text.splitlines()]
    chunks, buf, size = [], [], 0
    for p in paras:
        if not p:
            if buf:
                chunks.append("\n".join(buf))
                buf = []
                size = 0
            continue
        if size + len(p) + 1 > max_chunk_chars and buf:
            chunks.append("\n".join(buf))
            buf = []
            size = 0
        buf.append(p)
        size += len(p) + 1
    if buf:
        chunks.append("\n".join(buf))

    def score(s: str) -> int:
        toks = set(re.findall(r"[A-Za-z0-9]{3,}", s.lower()))
        base = sum(1 for t in toks if t in q_tokens)
        bonus = sum(3 for rx in cue_res if rx.search(s))
        nums = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:m|mL|min|°C|h|pH)\b", s))
        return base + bonus + nums

    ranked = sorted(chunks, key=score, reverse=True)
    return ranked[:top_k]


def pick_method_paragraph(text: str) -> str:
    text = normalize_pdf_text(text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return ""

    cues = [
        r"\b0\.1\s*m\b",
        r"\b0\.45\s*m\b",
        r"\b100\s*m[lL]\b",
        r"\b5\s*m[lL]\s*/?\s*min\b",
        r"\b45\s*°\s*C\b",
        r"\b50\s*°\s*C\b",
        r"\b24\s*h\b",
        r"\bpH\s*12\b",
        r"\bautotitrator\b|\bdl50\b|\bmettler\b",
        r"\bwater bath\b",
        r"\bfiltered?\b|\bvacuum\b",
        r"\bdry(?:ed|ing)?\b",
    ]
    rx = [re.compile(c, re.I) for c in cues]

    def sc(p: str) -> int:
        return sum(1 for r in rx if r.search(p)) + len(re.findall(r"\b\d", p))

    best = max(paragraphs, key=sc)
    return best
