import re, math
from typing import List, Dict, Any, Tuple, Iterable, Optional

_WORD_RX = re.compile(r"[A-Za-z0-9\-\u00C0-\u024F]+")
CHEM_RX = re.compile(r"\b(?:[A-Z][a-z]?[\d_]*){1,}\b")  # crude formula detector: SiO2, Al2O3, TiO2, etc.

# Morphology & process vocab (generic)
MORPHOLOGY_SYNONYMS = {
    "nanowire": {"nanowire","nanowires","nw","nws","nanofiber","nanofibers","nanorod","nanorods","nanowhisker","nanowhiskers"},
    "nanotube": {"nanotube","nanotubes","cnt","cnts"},
    "nanoparticle": {"nanoparticle","nanoparticles","np","nps","quantum dot","quantum dots","qd","qds"},
    "nanosheet": {"nanosheet","nanosheets","nanosheeted","2d"},
}
PROCESS_SYNONYMS = {
    "synthesis","synthesize","prepare","preparation","growth","fabrication","route","protocol",
    "sol-gel","hydrothermal","solvothermal","microwave","spray pyrolysis","cvd","pvd","vls","oxide-assisted",
    "electrospinning","anodic","template","calcination","anneal","annealing","seeded","etch","etching"
}
# Broad negatives for "on a substrate" (not the target material itself)
NEGATIVE_SUBSTRATE_CUES = {
    "on ", "substrate", "interconnect metallization", "graphene", "mos2", "gan", "ws2", "wse2", "hbn",
    "sapphire","mica","glass","quartz","silica","sio2","silicon wafer","si wafer"
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _tok(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RX.findall(s or "")]

def _contains_any(text: str, terms: Iterable[str]) -> bool:
    L = text.lower()
    return any(t in L for t in terms)

# ---------- spaCy-powered term extraction (optional) ----------
def _load_spacy() -> Optional[object]:
    try:
        import spacy
        for name in ("en_core_web_sm",):
            try:
                return spacy.load(name, disable=["parser","tagger","lemmatizer"])
            except Exception:
                pass
        return None
    except Exception:
        return None

_NLP = None  # lazy

def extract_query_profile(question: str) -> Dict[str, set]:
    """Extract target MATERIAL(s), MORPHOLOGY term(s), and PROCESS cue(s) from the question."""
    global _NLP
    q = _norm(question)
    materials, morphs, procs = set(), set(), set()

    # Try spaCy NER if available & you trained custom labels
    try:
        if _NLP is None:
            _NLP = _load_spacy()
        if _NLP is not None:
            doc = _NLP(q)
            for ent in getattr(doc, "ents", []):
                label = (ent.label_ or "").upper()
                txt = ent.text.strip()
                if not txt: 
                    continue
                if label in {"MATERIAL","CHEM","CHEMICAL","COMPOUND"}:
                    materials.add(txt.lower())
                elif label in {"MORPHOLOGY","SHAPE","FORM"}:
                    morphs.add(txt.lower())
                elif label in {"PROCESS","SYNTHESIS"}:
                    procs.add(txt.lower())
    except Exception:
        pass

    # Heuristics: chemical formula & common words
    for m in CHEM_RX.findall(q):
        # Filter silly matches like single letters
        if len(m) >= 3 and any(c.isdigit() for c in m):
            materials.add(m.lower())
    # common material words present in question
    for t in _tok(q):
        if t in {"silica","quartz","silicon","alumina","titania","zno","gan","mos2","ws2","hbn"}:
            materials.add(t)

    # Morphology synonyms from question
    qlow = q.lower()
    for key, syns in MORPHOLOGY_SYNONYMS.items():
        if _contains_any(qlow, syns):
            morphs.update(syns)

    # Process cues from question
    for w in PROCESS_SYNONYMS:
        if w in qlow:
            procs.add(w)

    return {"materials": materials, "morphology": morphs, "process": procs}

def sanitize_authors_field(ref: Dict[str, Any]) -> None:
    a = ref.get("authors")
    if isinstance(a, list):
        if a and all(isinstance(x, str) and len(_tok(x)) <= 2 for x in a):
            ref["authors"] = None
    elif isinstance(a, str):
        if re.search(r"(,\s*){3,}", a):
            ref["authors"] = re.sub(r"(?:\s*,\s*)+", " ", a).strip()

def extract_used_ref_indexes(answer: str) -> List[int]:
    if not answer: return []
    idxs = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        try:
            n = int(m.group(1))
            if n >= 1: idxs.add(n)
        except ValueError:
            pass
    return sorted(idxs)

def _score_ref_against_profile(ref_text: str, profile: Dict[str, set]) -> float:
    """Weighted token overlap + bonuses for profile matches; penalty for substrate contexts."""
    rtoks = set(_tok(ref_text))
    base = 0.0
    # Generic overlap with all profile tokens
    all_profile = set().union(profile.get("materials", set()),
                              profile.get("morphology", set()),
                              profile.get("process", set()))
    base += len(rtoks & all_profile) / (1 + math.log(1 + len(rtoks)))

    # Bonuses
    text = ref_text.lower()
    # Strong bonus if any exact material mention (formula or name)
    if any(m in text for m in profile.get("materials", ())): base += 0.7
    # Morphology match
    if any(m in text for m in profile.get("morphology", ())): base += 0.5
    # Process cue
    if any(p in text for p in profile.get("process", ())): base += 0.25

    # Penalize classic substrate phrases (likely “X on Y” not “Y nanowires”)
    if _contains_any(text, NEGATIVE_SUBSTRATE_CUES): base -= 0.6

    return base

def select_references(answer: str, refs: List[Dict[str, Any]], *, question: Optional[str], top_k: int = 6) -> Tuple[List[Dict[str, Any]], List[int]]:
    if not refs: return [], []

    for r in refs: sanitize_authors_field(r)

    # If the model produced explicit [n] markers, honor those.
    used = extract_used_ref_indexes(answer)
    if used:
        selected = []
        for n in used:
            i = n - 1
            if 0 <= i < len(refs):
                selected.append(refs[i])
        # Pad to at least 2 if needed by ranking others against question
        if question and len(selected) < min(2, top_k):
            prof = extract_query_profile(question)
            pool = [(i, r) for i, r in enumerate(refs) if (i+1) not in used]
            scored = sorted(pool, key=lambda ir: _score_ref_against_profile(
                _norm(ir[1].get("title") or ir[1].get("citation") or ir[1].get("name") or "") + " " +
                _norm(ir[1].get("meta") or ir[1].get("journal") or ir[1].get("source") or ""), prof
            ), reverse=True)
            for i, r in scored:
                if len(selected) >= top_k: break
                selected.append(r); used.append(i+1)
        return selected, sorted(set(used))

    # No [n] markers → rank all against the query profile
    prof = extract_query_profile(question or "")
    scored_all = sorted(
        refs,
        key=lambda r: _score_ref_against_profile(
            _norm(r.get("title") or r.get("citation") or r.get("name") or "") + " " +
            _norm(r.get("meta") or r.get("journal") or r.get("source") or ""), prof
        ),
        reverse=True
    )
    chosen = scored_all[:top_k]
    return chosen, list(range(1, min(top_k, len(chosen)) + 1))

def build_references_payload(answer: str, refs_full: List[Dict[str, Any]], top_k: int = 6, *, question: Optional[str] = None) -> Dict[str, Any]:
    chosen, used = select_references(answer, refs_full, question=question, top_k=top_k)

    # Build a simple ACS block from chosen
    lines = []
    for i, r in enumerate(chosen, 1):
        title = _norm(r.get("title") or r.get("citation") or r.get("name") or f"Reference {i}")
        journal = _norm(r.get("journal") or r.get("source") or r.get("meta") or "")
        year = str(r.get("year") or "")
        doi  = _norm(r.get("doi") or "")
        parts = [title]
        if journal: parts.append(journal)
        if year:    parts.append(year)
        if doi:     parts.append(doi)
        lines.append(f"{i}. " + ", ".join([p for p in parts if p]))
    block = "\n".join(lines)

    return {"references": chosen, "used_ref_indexes": used, "references_block": block}
