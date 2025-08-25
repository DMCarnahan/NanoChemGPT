import re, math
from typing import List, Dict, Any, Tuple, Iterable

_WORD_RX = re.compile(r"[A-Za-z0-9\-\u00C0-\u024F]+")

SIO2_SYNONYMS = {"sio2","silica","silicon dioxide","amorphous silica","quartz"}
NW_SYNONYMS = {"nanowire","nanowires","nw","nws","nanofiber","nanofibers","nanorod","nanorods","nanowhisker","nanowhiskers"}
SYNTHESIS_SYNONYMS = {"synthesis","prepare","preparation","growth","fabrication","sol-gel","hydrothermal","solvothermal","cvd","vls","vapour","vapor","oxide-assisted","electrospinning","anneal","calcination","template","anodic","aao"}
NEGATIVE_SUBSTRATE_CUES = {"on sio2","on silicon dioxide","on silica","graphene","mos2","gan","ws2","wse2","hbn","interconnect metallization","mems gas sensors","micro-nanorobots"}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _tok(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RX.findall(s or "")]

def _contains_any(text: str, terms: Iterable[str]) -> bool:
    L = text.lower()
    return any(t in L for t in terms)

def coerce_ref(r: Any) -> Dict[str, Any]:
    """Make every ref look like a dict with at least a 'title'."""
    if isinstance(r, dict):
        return r
    if isinstance(r, str):
        return {"title": _norm(r)}
    return {"title": _norm(str(r))}

def is_probably_about_sio2_nws(ref: Dict[str, Any]) -> bool:
    title = _norm(ref.get("title") or ref.get("citation") or ref.get("name") or "")
    meta  = _norm(ref.get("meta") or ref.get("journal") or ref.get("source") or "")
    text  = f"{title} {meta}".lower()
    if not _contains_any(text, SIO2_SYNONYMS): 
        return False
    if not _contains_any(text, NW_SYNONYMS):
        return False
    if not _contains_any(text, SYNTHESIS_SYNONYMS):
        if "first-principles" not in text and "density functional" not in text:
            return False
    if _contains_any(text, NEGATIVE_SUBSTRATE_CUES):
        if "silica nanowire" not in text and "silicon dioxide nanowire" not in text:
            return False
    return True

def sanitize_authors_field(ref: Dict[str, Any]) -> None:
    a = ref.get("authors")
    if isinstance(a, list):
        if a and all(isinstance(x, str) and len(_tok(x)) <= 2 for x in a):
            ref["authors"] = None
    elif isinstance(a, str):
        if re.search(r"(,\s*){3,}", a):
            ref["authors"] = re.sub(r"(?:\s*,\s*)+", " ", a).strip()

def extract_used_ref_indexes(answer: str) -> List[int]:
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
    rtoks = set(_tok(ref_text))
    ttoks = set(_tok(text))
    inter = rtoks & ttoks
    base = len(inter) / (1 + (len(rtoks) and __import__("math").log(1 + len(rtoks))))
    bonus = 0.0
    if _contains_any(ref_text, SIO2_SYNONYMS): bonus += 0.5
    if _contains_any(ref_text, NW_SYNONYMS):  bonus += 0.5
    if _contains_any(ref_text, SYNTHESIS_SYNONYMS): bonus += 0.25
    if _contains_any(ref_text, NEGATIVE_SUBSTRATE_CUES): bonus -= 0.6
    return float(base + bonus)

def _text_of(ref: Dict[str, Any]) -> str:
    return _norm(ref.get("title") or ref.get("citation") or ref.get("name") or "") + " " + _norm(ref.get("meta") or ref.get("journal") or ref.get("source") or "")

def select_references(answer: str, refs_raw: List[Any], top_k: int = 6) -> Tuple[List[Dict[str, Any]], List[int]]:
    if not refs_raw:
        return [], []
    refs = [coerce_ref(r) for r in refs_raw]
    for r in refs:
        sanitize_authors_field(r)

    used = extract_used_ref_indexes(answer)
    if used:
        selected = []
        for n in used:
            i = n - 1
            if 0 <= i < len(refs):
                selected.append(refs[i])
        if len(selected) < min(2, top_k):
            pool = [(i, r) for i, r in enumerate(refs) if (i+1) not in used]
            scored = sorted(pool, key=lambda ir: _score_ref_against_text(_text_of(ir[1]), answer), reverse=True)
            for i, r in scored:
                if len(selected) >= top_k: break
                selected.append(r)
                used.append(i+1)
        return selected, sorted(set(used))

    filtered = [r for r in refs if is_probably_about_sio2_nws(r)]
    if not filtered:
        filtered = refs  # fallback: keep something
    scored = sorted(filtered, key=lambda r: _score_ref_against_text(_text_of(r), answer), reverse=True)
    chosen = scored[:max(1, min(top_k, len(scored)))]
    return chosen, list(range(1, len(chosen)+1))

def build_references_payload(answer: str, refs_full: List[Any], top_k: int = 6) -> Dict[str, Any]:
    chosen, used = select_references(answer, refs_full, top_k=top_k)
    lines = []
    for i, r in enumerate(chosen, 1):
        title = _norm(r.get("title") or r.get("citation") or r.get("name") or f"Reference {i}")
        journal = _norm(r.get("journal") or r.get("source") or r.get("meta") or "")
        year = str(r.get("year") or "")
        doi  = _norm(r.get("doi") or "")
        parts = [title] + [p for p in (journal, year, doi) if p]
        lines.append(f"{i}. " + ", ".join(parts))
    return {
        "references": chosen,
        "used_ref_indexes": used,
        "references_block": "\n".join(lines)
    }
