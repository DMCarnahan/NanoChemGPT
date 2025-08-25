import re
import math
from typing import List, Dict, Any, Tuple, Iterable

_WORD_RX = re.compile(r"[A-Za-z0-9\-\u00C0-\u024F]+")

SIO2_SYNONYMS = {
    "sio2", "silica", "silicon dioxide", "amorphous silica", "quartz"
}
NW_SYNONYMS = {
    "nanowire", "nanowires", "nw", "nws", "nanofiber", "nanofibers", "nanorod", "nanorods", "nanowhisker", "nanowhiskers"
}
SYNTHESIS_SYNONYMS = {
    "synthesis", "prepare", "preparation", "growth", "fabrication", "sol-gel",
    "hydrothermal", "solvothermal", "cvd", "vls", "vapour", "vapor", "oxide-assisted",
    "electrospinning", "anneal", "calcination", "template", "anodic", "aaO"
}

# Terms that frequently indicate the paper is about *something on SiO2 substrate* (not SiO2 NWs)
NEGATIVE_SUBSTRATE_CUES = {
    "on sio2", "on silicon dioxide", "on silica", "graphene", "mos2", "gan", "ws2", "wse2", "hbn",
    "interconnect metallization", "mems gas sensors", "micro-nanorobots",
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _tok(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RX.findall(s or "")]

def _contains_any(text: str, terms: Iterable[str]) -> bool:
    L = text.lower()
    return any(t in L for t in terms)

def expand_query_synonyms(q: str) -> List[str]:
    qn = _norm(q).lower()
    bag = set(_tok(qn))
    if "sio2" in bag or "silica" in bag or "silicon" in bag:
        bag.update(SIO2_SYNONYMS)
    if "nanowire" in bag or "nanowires" in bag or "nanofiber" in bag or "nanorod" in bag:
        bag.update(NW_SYNONYMS)
    if _contains_any(qn, ("synthesis","prepare","preparation","growth","fabrication", "procedure")):
        bag.update(SYNTHESIS_SYNONYMS)
    return sorted(bag)

def is_probably_about_sio2_nws(ref: Dict[str, Any]) -> bool:
    """Heuristically keep refs actually about SiO2 NWs, not 'X on SiO2'."""
    title = _norm(ref.get("title") or ref.get("citation") or ref.get("name") or "")
    meta  = _norm(ref.get("meta") or ref.get("journal") or ref.get("source") or "")
    text  = f"{title} {meta}".lower()
    # Must have SiO2/silica AND NW keywords
    if not _contains_any(text, SIO2_SYNONYMS): 
        return False
    if not _contains_any(text, NW_SYNONYMS):
        return False
    # Should include a synthesis/growth cue
    if not _contains_any(text, SYNTHESIS_SYNONYMS):
        # allow purely structural SiO2 nanowires (first-principles) as weak positive
        if "first-principles" not in text and "density functional" not in text:
            return False
    # Exclude obvious "on SiO2" substrate contexts
    if _contains_any(text, NEGATIVE_SUBSTRATE_CUES):
        # allow explicit "silica nanowires" wording to override weak negatives
        if "silica nanowire" not in text and "silicon dioxide nanowire" not in text:
            return False
    return True

def sanitize_authors_field(ref: Dict[str, Any]) -> None:
    """Fix cases where authors are split like 'W, a, n, g'."""
    a = ref.get("authors")
    if isinstance(a, list):
        # If list items are single letters or contain many commas, join smartly
        if a and all(isinstance(x, str) and len(_tok(x)) <= 2 for x in a):
            # Probably char-split; drop it
            ref["authors"] = None
    elif isinstance(a, str):
        # If string looks like "W, a, n, g", collapse
        if re.search(r"(,\s*){3,}", a):
            ref["authors"] = re.sub(r"(?:\s*,\s*)+", " ", a).strip()
    # else leave as-is

def extract_used_ref_indexes(answer: str) -> List[int]:
    """Return sorted unique 1-based indexes from [1], [2] markers in answer."""
    if not answer:
        return []
    idxs = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        try:
            n = int(m.group(1))
            if n >= 1:
                idxs.add(n)
        except ValueError:
            pass
    return sorted(idxs)

def _score_ref_against_text(ref_text: str, text: str) -> float:
    """Very simple relevance: weighted token overlap + cues."""
    rtoks = set(_tok(ref_text))
    ttoks = set(_tok(text))
    inter = rtoks & ttoks
    base = len(inter) / (1 + math.log(1 + len(rtoks)))
    bonus = 0.0
    if _contains_any(ref_text, SIO2_SYNONYMS): bonus += 0.5
    if _contains_any(ref_text, NW_SYNONYMS): bonus += 0.5
    if _contains_any(ref_text, SYNTHESIS_SYNONYMS): bonus += 0.25
    if _contains_any(ref_text, NEGATIVE_SUBSTRATE_CUES): bonus -= 0.6
    return base + bonus

def select_references(answer: str, refs: List[Dict[str, Any]], top_k: int = 6) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Decide which references to display.
    - If [n] markers exist: return exactly those, in numeric order intersected with available refs.
    - Else: filter + score and return top_k most relevant to SiO2 NW synthesis.
    Returns (selected_refs, used_ref_indexes_1based)
    """
    if not refs:
        return [], []

    # Sanitize authors fields in-place
    for r in refs:
        sanitize_authors_field(r)

    used = extract_used_ref_indexes(answer)
    if used:
        # Keep intersection within bounds
        selected = []
        for n in used:
            i = n - 1
            if 0 <= i < len(refs):
                selected.append(refs[i])
        # If too few (e.g., only 1), pad with best matches (but don't exceed top_k)
        if len(selected) < min(2, top_k):
            # score all not already selected
            pool = [(i, r) for i, r in enumerate(refs) if (i+1) not in used]
            scored = sorted(pool, key=lambda ir: _score_ref_against_text(
                _norm(ir[1].get("title") or ir[1].get("citation") or ir[1].get("name") or "") + " " + _norm(ir[1].get("meta") or ir[1].get("journal") or ir[1].get("source") or ""),
                answer
            ), reverse=True)
            for i, r in scored:
                if len(selected) >= top_k:
                    break
                selected.append(r)
                used.append(i+1)
        return selected, sorted(set(used))

    # No explicit markers → filter + score
    filtered = [r for r in refs if is_probably_about_sio2_nws(r)]
    if not filtered:
        filtered = refs  # fallback to all

    scored = sorted(
        enumerate(filtered),
        key=lambda ir: _score_ref_against_text(
            _norm(ir[1].get("title") or ir[1].get("citation") or ir[1].get("name") or "") + " " + _norm(ir[1].get("meta") or ir[1].get("journal") or ir[1].get("source") or ""),
            answer
        ),
        reverse=True
    )
    # Map back to original indexes if filtered came from subset
    selected = [ir[1] for ir in scored[:top_k]]
    # We don't know the original 1-based positions reliably here (due to filtering from unknown original order).
    # For frontend display, it's fine to omit used_ref_indexes or set to 1..len(selected).
    return selected, list(range(1, min(top_k, len(selected)) + 1))

def build_references_payload(answer: str, refs_full: List[Dict[str, Any]], top_k: int = 6) -> Dict[str, Any]:
    """
    Returns a dict ready to jsonify:
    { "references": [...], "used_ref_indexes": [..], "references_block": "..." }
    """
    chosen, used = select_references(answer, refs_full, top_k=top_k)

    # Build a simple block (ACS-ish) from chosen
    lines = []
    for i, r in enumerate(chosen, 1):
        title = _norm(r.get("title") or r.get("citation") or r.get("name") or f"Reference {i}")
        journal = _norm(r.get("journal") or r.get("source") or r.get("meta") or "")
        year = str(r.get("year") or "")
        doi  = _norm(r.get("doi") or "")
        line_parts = [title]
        if journal: line_parts.append(journal)
        if year: line_parts.append(year)
        if doi: line_parts.append(doi)
        lines.append(f"{i}. " + ", ".join([p for p in line_parts if p]))
    block = "\n".join(lines)

    return {
        "references": chosen,
        "used_ref_indexes": used,
        "references_block": block
    }
