import os, re, math
from typing import List, Dict, Any, Optional, Tuple

# --------- small helpers ---------
_WORD_RX = re.compile(r"[A-Za-z0-9\-\u00C0-\u024F]+")
CHEM_RX  = re.compile(r"\b(?:[A-Z][a-z]?[\d_]*){2,}\b")  # generic formula (e.g., SiO2, Al2O3, MoS2)

def _tok(s: str) -> List[str]: return [t.lower() for t in _WORD_RX.findall(s or "")]
def _norm(s: str) -> str: return re.sub(r"\s+", " ", (s or "").strip())

def sanitize_authors_field(ref: Dict[str, Any]) -> None:
    a = ref.get("authors")
    if isinstance(a, list) and a and all(isinstance(x, str) and len(_tok(x)) <= 2 for x in a):
        ref["authors"] = None

def extract_used_ref_indexes(answer: str) -> List[int]:
    out = set()
    for m in re.finditer(r"\[(\d+)\]", answer or ""):
        try: out.add(int(m.group(1)))
        except: pass
    return sorted(out)

# --------- spaCy pipeline ---------
_SPACY_MODEL = os.getenv("SPACY_MODEL")  
_NLP = None
def _nlp():
    global _NLP
    if _NLP is not None: return _NLP
    try:
        import spacy
        if _SPACY_MODEL:
            _NLP = spacy.load(_SPACY_MODEL)
        else:
            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except Exception:
        _NLP = None
    return _NLP

# Label sets — adjust if your model uses different names
MATERIAL_LABELS = {"MATERIAL","CHEM","CHEMICAL","COMPOUND"}
MORPH_LABELS    = {"MORPHOLOGY","SHAPE","FORM"}
PROCESS_LABELS  = {"PROCESS","SYNTHESIS","METHOD"}

# generic “nano-ness” detector (no long synonym lists)
NANO_RX = re.compile(r"\bnano[\w-]*\b", re.I)
QDOT_RX = re.compile(r"\bquantum\s+dots?\b", re.I)  # special case outside “nano*”

# --------- substrate heuristics (tunable) ---------
SUBSTRATE_PREPS = {"on", "onto", "over", "above"}
SUBSTRATE_VERBS = {
    "grow","grown","deposit","deposited","coat","coated","form","formed",
    "assemble","assembled","print","printed","pattern","patterned",
    "evaporate","evaporated","sputter","sputtered","adsorb","adsorbed",
    "support","supported","place","placed","transfer","transferred"
}
SUBSTRATE_WORDS = {"substrate","wafer","support","template","seed-layer","seed","mica","sapphire","glass"}
_SUBSTRATE_PENALTY = float(os.getenv("REF_SUBSTRATE_PENALTY", "0.75"))  # stronger default
_SUBSTRATE_STRICT = os.getenv("REF_SUBSTRATE_STRICT", "1") not in {"0","false","False"}

def _extract_query_profile(question: str) -> Dict[str, set]:
    """Use your spaCy NER (if present) to pull MATERIAL / MORPHOLOGY / PROCESS.
       Fallbacks: chemical formulas; simple 'nano*' and 'quantum dot(s)' cues."""
    q = _norm(question)
    mats, morphs, procs = set(), set(), set()

    nlp = _nlp()
    if nlp:
        doc = nlp(q)
        for ent in getattr(doc, "ents", []):
            L = (ent.label_ or "").upper()
            if L in MATERIAL_LABELS: mats.add(ent.text.lower())
            elif L in MORPH_LABELS:  morphs.add(ent.text.lower())
            elif L in PROCESS_LABELS: procs.add(ent.text.lower())
        # if the model didn’t tag morphology, infer generic nanomorph cues from tokens
        if not morphs:
            text = doc.text
            if NANO_RX.search(text): morphs.add("nano*")
            if QDOT_RX.search(text): morphs.add("quantum dot")
    else:
        # minimal fallback without spaCy
        for m in CHEM_RX.findall(q): mats.add(m.lower())
        if NANO_RX.search(q): morphs.add("nano*")
        if QDOT_RX.search(q): morphs.add("quantum dot")
        for w in ("synthesis","prepare","preparation","growth","fabrication","sol-gel","hydrothermal","solvothermal","cvd","pvd","anneal","calcination"):
            if w in q.lower(): procs.add(w)

    # always add formula-like tokens from the question
    for m in CHEM_RX.findall(q):
        mats.add(m.lower())

    return {"materials": mats, "morphology": morphs, "process": procs}

def _doc_matches_target(txt: str, profile: Dict[str, set]) -> Tuple[bool, bool]:
    """
    Returns (match_ok, substrate_like).
    match_ok: text looks about the requested material/morphology (or generic 'nano*')
    substrate_like: text likely uses the material as a substrate (e.g., 'X grown on SiO2')
    """
    t = _norm(txt)
    nlp = _nlp()
    mats = set(m.lower() for m in profile.get("materials") or [])
    morphs = set(m.lower() for m in profile.get("morphology") or [])

    # trivial quick checks
    has_nano = bool(NANO_RX.search(t) or QDOT_RX.search(t))
    has_mat_mention = any(m in t.lower() for m in mats) if mats else True  # allow if no explicit material asked

    if not nlp:
        # fallback: if a material was requested, require it; then require 'nano*' if no morphology given
        if mats and not has_mat_mention: return (False, False)
        if not morphs and not has_nano:  return (False, False)
        # fallback substrate heuristic (harsher): verbs + preps + 'on <mat> (substrate|wafer|template)'
        if mats:
            mat_pat = "|".join(map(re.escape, mats))
            sub_rx = re.compile(
                rf"(?:\\b(?:{'|'.join(SUBSTRATE_VERBS)})\\b\\s+)?\\b(?:{'|'.join(SUBSTRATE_PREPS)})\\s+(?:{mat_pat})(?:\\s+(?:{'|'.join(SUBSTRATE_WORDS)}))?\\b",
                re.I
            )
            substrate_like = bool(sub_rx.search(t))
        else:
            substrate_like = False
        # In fallback mode, if the only mention is substrate-like and no morphology near material, block
        if substrate_like and _SUBSTRATE_STRICT:
            return (False, True)
        return (True, substrate_like)

    # spaCy path: check proximity between material entities and morphology cues
    doc = nlp(t)

    def _substrate_context(ent) -> bool:
        # pobj of on/onto/over
        for tok in ent:
            if tok.dep_ == "pobj" and tok.head.lemma_.lower() in SUBSTRATE_PREPS:
                return True
        # verb ... on <ent>
        for tok in doc:
            if tok.pos_ == "VERB" and tok.lemma_.lower() in SUBSTRATE_VERBS:
                for child in tok.children:
                    if child.lemma_.lower() in SUBSTRATE_PREPS:
                        pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
                        if pobj and ent.start <= pobj.i < ent.end:
                            return True
        # "on <ent> substrate/wafer/template"
        after_i = ent[-1].i + 1
        if after_i < len(doc) and doc[after_i].lemma_.lower() in SUBSTRATE_WORDS:
            # ensure there is an 'on/onto/over' governing this span
            for tok in ent:
                if tok.dep_ == "pobj" and tok.head.lemma_.lower() in SUBSTRATE_PREPS:
                    return True
        return False
    mat_spans = []
    for ent in getattr(doc, "ents", []):
        if (ent.label_ or "").upper() in MATERIAL_LABELS:
            mat_spans.append(ent)
        elif mats and any(m in ent.text.lower() for m in mats):
            mat_spans.append(ent)

    if not mats:
        if morphs:
            # look for any token span overlapping a morph term from profile
            has_morph_from_profile = any(m in t.lower() for m in morphs) or has_nano
            return (has_morph_from_profile, False)
        return (has_nano, False)

    # Require at least one material mention when a material was asked
    if mats and not (mat_spans or has_mat_mention):
        return (False, False)

    # proximity: any “nano*” or morphology term within +/- 6 tokens of a material entity
    near = False
    substrate_like = False
    substrate_mentions = 0

    # collect token positions where “nano*” or morph terms appear
    morph_pos = set()
    for i, tok in enumerate(doc):
        L = tok.text.lower()
        if NANO_RX.match(L) or QDOT_RX.match(doc.text[max(0, tok.idx-8): tok.idx+len(tok)]):
            morph_pos.add(i)
        if morphs and any(m in L for m in morphs):
            morph_pos.add(i)

    # Evaluate proximity and “on <material>” prepositional objects
    for ent in mat_spans:
        for i in morph_pos:
            if abs(i - ent.start) <= 6 or abs(i - ent.end) <= 6:
                near = True
        # substrate grammar checks
        if _substrate_context(ent):
            substrate_like = True
            substrate_mentions += 1

    # If every mention is substrate-like and there is no morphology cue near the material, block it outright
    if mat_spans and substrate_mentions == len(mat_spans) and not near and _SUBSTRATE_STRICT:
        return (False, True)

    if morphs and not near:
        return (False, substrate_like)

    if (near or has_nano or bool(morphs)) and (mat_spans or has_mat_mention):
        return (True, substrate_like)

    return (False, substrate_like)

def _score_ref(txt: str, profile: Dict[str, set]) -> float:
    """Rank by entity/cue overlap + optional vector similarity; mild penalty for pure substrate context."""
    L = txt.lower()
    base = 0.0
    if any(m in L for m in profile.get("materials", set())): base += 0.7
    if any(p in L for p in profile.get("process", set())):   base += 0.3
    if NANO_RX.search(L) or QDOT_RX.search(L):               base += 0.3
    if profile.get("morphology"):
        if any(m in L for m in profile["morphology"]):       base += 0.3

    nlp = _nlp()
    if nlp and nlp.vocab.vectors_length:
        try:
            qtxt = " ".join(list(profile.get("materials", set()) |
                                 profile.get("morphology", set()) |
                                 profile.get("process", set())))
            if qtxt.strip():
                qdoc = nlp(_norm(qtxt))
                tdoc = nlp(_norm(txt))
                base += 0.5 * float(qdoc.similarity(tdoc))
        except Exception:
            pass

    # penalize explicit substrate phrasing (harsher, tunable)
    mats = profile.get("materials", set())
    if mats:
        mat_pat = "|".join(map(re.escape, mats))
        sub_rx = re.compile(
            rf"(?:\\b(?:{'|'.join(SUBSTRATE_VERBS)})\\b\\s+)?\\b(?:{'|'.join(SUBSTRATE_PREPS)})\\s+(?:{mat_pat})(?:\\s+(?:{'|'.join(SUBSTRATE_WORDS)}))?\\b",
            re.I
        )
        if sub_rx.search(L):
            base -= _SUBSTRATE_PENALTY
    return base

# --------- main selection API ---------
def select_references(answer: str, refs: List[Dict[str, Any]], *, question: Optional[str], top_k: int = 6) -> Tuple[List[Dict[str, Any]], List[int]]:
    if not refs: return [], []

    for r in refs: sanitize_authors_field(r)

    # If the answer has explicit [n] citations, honor them exactly
    used = extract_used_ref_indexes(answer)
    if used:
        chosen = []
        for n in used:
            i = n - 1
            if 0 <= i < len(refs): chosen.append(refs[i])
        return chosen[:top_k], used[:top_k]

    # Otherwise: spaCy-driven gating + ranking
    prof = _extract_query_profile(question or "")

    gated = []
    for r in refs:
        title = _norm(r.get("title") or r.get("citation") or r.get("name") or "")
        meta  = _norm(r.get("meta") or r.get("journal") or r.get("source") or "")
        ok, substrate_like = _doc_matches_target(f"{title} {meta}", prof)
        if ok:
            # stronger downweight if substrate phrasing survived gating
            r["_substrate_penalty"] = _SUBSTRATE_PENALTY if substrate_like else 0.0
            gated.append(r)

    # Relax gate if too few candidates 
    pool = gated if len(gated) >= max(2, top_k//2) else list(refs)

    scored = []
    for r in pool:
        title = _norm(r.get("title") or r.get("citation") or r.get("name") or "")
        meta  = _norm(r.get("meta") or r.get("journal") or r.get("source") or "")
        s = _score_ref(f"{title} {meta}", prof) - float(r.get("_substrate_penalty", 0.0))
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    chosen = [r for _, r in scored[:top_k]]
    return chosen, list(range(1, min(top_k, len(chosen)) + 1))

def build_references_payload(answer: str, refs_full: List[Dict[str, Any]], top_k: int = 6, *, question: Optional[str] = None) -> Dict[str, Any]:
    chosen, used = select_references(answer, refs_full, question=question, top_k=top_k)
    lines = []
    for i, r in enumerate(chosen, 1):
        title = _norm(r.get("title") or r.get("citation") or r.get("name") or f"Reference {i}")
        journal = _norm(r.get("journal") or r.get("source") or r.get("meta") or "")
        year = str(r.get("year") or "")
        doi  = _norm(r.get("doi") or "")
        line = f"{i}. " + ", ".join([p for p in (title, journal, year, doi) if p])
        lines.append(line)
    return {"references": chosen, "used_ref_indexes": used, "references_block": "\n".join(lines)}
