from __future__ import annotations

import math
import re
import unicodedata
from html import unescape as _html_unescape
from typing import Dict, List, Optional, Set, Tuple

# ------------------------
# Text utilities
# ------------------------

_WS_RX = re.compile(r"\\s+")
_PUNCT_RX = re.compile(
    r"[\\u2010\\u2011\\u2012\\u2013\\u2014\\u2015\\-\\p{P}]", re.UNICODE
)


def _lower(s: Optional[str]) -> str:
    return s.lower() if isinstance(s, str) else ""


def _strip(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def _norm_space(s: str) -> str:
    return _WS_RX.sub(" ", s).strip()


def _fold(s: str) -> str:
    # NFKD fold to strip accents; keep ASCII
    try:
        return (
            unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        )
    except Exception:
        return s


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = _html_unescape(s)
    s = _norm_space(s)
    return s


def _tokenize(s: str) -> List[str]:
    out = []
    token = []
    for ch in s:
        if ch.isalnum():
            token.append(ch.lower())
        else:
            if token:
                out.append("".join(token))
                token = []
    if token:
        out.append("".join(token))
    return out


def _shingles(s: str, k: int = 3) -> Set[str]:
    if len(s) < k:
        return {s} if s else set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}


def _jaccard_trigram(a: str, b: str) -> float:
    a = _fold(_clean_text(a.lower()))
    b = _fold(_clean_text(b.lower()))
    if not a or not b:
        return 0.0
    A = _shingles(a, 3)
    B = _shingles(b, 3)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


# ------------------------
# ID normalization
# ------------------------


def _norm_doi(x: Optional[str]) -> str:
    if not x:
        return ""
    s = _lower(x)
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    s = s.replace("doi:", "").strip()
    return s


def _norm_pmid(x: Optional[str]) -> str:
    if not x:
        return ""
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s


def _norm_arxiv(x: Optional[str]) -> str:
    if not x:
        return ""
    s = _lower(x).replace("arxiv:", "").replace("https://arxiv.org/abs/", "").strip()
    return s


def _title_key(title: Optional[str]) -> str:
    if not title:
        return ""
    s = _fold(_clean_text(title)).lower()
    # strip most punctuation and collapse whitespace
    s = re.sub(r"[^a-z0-9\\s]+", " ", s)
    s = _norm_space(s)
    return s


# ------------------------
# Citation parsing & renumbering
# ------------------------

# Matches [1], [1, 2, 5], [3–7], [3-7], and mixed combos like [2, 4–6, 9]
_CITE_RX = re.compile(r"\[(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\]")


def extract_used_ref_indexes(*texts: Optional[str]) -> List[int]:
    """
    Extract unique citation indexes from any number of strings.
    Supports single, comma, and range citations inside square brackets.
    Returns sorted unique list of ints.
    """
    used = set()
    for t in texts:
        if not t:
            continue
        for m in _CITE_RX.finditer(t):
            chunk = m.group(1)
            for part in re.split(r"\\s*,\\s*", chunk):
                if re.search(r"[-–]", part):
                    a, b = re.split(r"[-–]", part)
                    try:
                        a_i = int(a.strip())
                        b_i = int(b.strip())
                    except Exception:
                        continue
                    if a_i <= b_i:
                        for i in range(a_i, b_i + 1):
                            used.add(i)
                    else:
                        for i in range(b_i, a_i + 1):
                            used.add(i)
                else:
                    try:
                        used.add(int(part.strip()))
                    except Exception:
                        pass
    return sorted(used)


def renumber_citations(text: Optional[str], index_map: Dict[int, int]) -> str:
    """
    Rewrite bracketed citations using index_map (old->new). Keeps ranges when contiguous.
    """
    if not text:
        return ""

    def _rewrite(match: re.Match) -> str:
        raw = match.group(1)
        out = []
        for part in re.split(r"\\s*,\\s*", raw):
            if re.search(r"[-–]", part):
                a, b = re.split(r"[-–]", part)
                try:
                    a_i = int(a.strip())
                    b_i = int(b.strip())
                except Exception:
                    continue
                lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
                mapped = [index_map.get(i, i) for i in range(lo, hi + 1)]
                # Compress if contiguous
                if mapped and mapped == list(range(mapped[0], mapped[0] + len(mapped))):
                    out.append(f"{mapped[0]}–{mapped[-1]}")
                else:
                    out.extend(str(x) for x in mapped)
            else:
                try:
                    i = int(part.strip())
                    out.append(str(index_map.get(i, i)))
                except Exception:
                    pass
        return "[" + ", ".join(out) + "]"

    return _CITE_RX.sub(_rewrite, text)


# ------------------------
# Dedupe
# ------------------------


def _choose_primary(a: dict, b: dict) -> dict:
    """Heuristic: prefer entry with DOI; else longer abstract; else has year; else 'a'."""
    a_doi = bool(_norm_doi(a.get("doi")))
    b_doi = bool(_norm_doi(b.get("doi")))
    if a_doi != b_doi:
        return a if a_doi else b
    a_abs = len(_clean_text(a.get("abstract", "")))
    b_abs = len(_clean_text(b.get("abstract", "")))
    if a_abs != b_abs:
        return a if a_abs > b_abs else b
    a_year = _strip(str(a.get("year", "")))
    b_year = _strip(str(b.get("year", "")))
    if bool(a_year) != bool(b_year):
        return a if a_year else b
    return a  # stable


def dedupe_refs(
    refs: List[dict], title_sim_threshold: float = 0.85
) -> Tuple[List[dict], Dict[int, int], List[List[int]]]:
    """
    Deduplicate by DOI/PMID/arXiv; else fuzzy by title (trigram Jaccard).
    Returns:
      unique_refs: list with preserved essential fields; assigned 'index' if missing.
      merge_map:   {old_index -> kept_index}
      groups:      list of groups of merged original indexes
    """
    # ensure every ref has an 'index' (1-based)
    for i, r in enumerate(refs, 1):
        if "index" not in r or r.get("index") is None:
            r["index"] = i

        # clean some fields
        r["title"] = _clean_text(r.get("title", ""))
        r["abstract"] = _clean_text(r.get("abstract", ""))
        if isinstance(r.get("authors"), list):
            r["authors"] = ", ".join(r["authors"])

    by_key = {}  # (doi|pmid|arxiv|titlekey) -> kept_ref
    kept = []
    merge_map: Dict[int, int] = {}
    groups: List[List[int]] = []

    # First pass: IDs (doi/pmid/arxiv) exact
    id_buckets: Dict[str, List[dict]] = {}
    for r in refs:
        keys = []
        doi = _norm_doi(r.get("doi"))
        pmid = _norm_pmid(r.get("pmid"))
        arx = _norm_arxiv(r.get("arxiv_id"))
        if doi:
            keys.append(("doi", doi))
        if pmid:
            keys.append(("pmid", pmid))
        if arx:
            keys.append(("arxiv", arx))
        if not keys:
            # temporary title key bucket
            tkey = _title_key(r.get("title"))
            if tkey:
                keys.append(("tkey", tkey))
        # use first available key to bucket (we'll fuzzy within title-buckets later)
        if keys:
            k = f"{keys[0][0]}:{keys[0][1]}"
        else:
            k = f"idx:{r['index']}"
        id_buckets.setdefault(k, []).append(r)

    # Resolve buckets
    for bucket in id_buckets.values():
        if len(bucket) == 1:
            r = bucket[0]
            kept.append(r)
            merge_map[r["index"]] = r["index"]
            groups.append([r["index"]])
        else:
            # prefer by IDs; if same title-key bucket, choose primary and map others
            base = bucket[0]
            for r in bucket[1:]:
                base = _choose_primary(base, r)
            kept.append(base)
            grp = []
            for r in bucket:
                merge_map[r["index"]] = base["index"]
                grp.append(r["index"])
            groups.append(sorted(set(grp)))

    # Second pass: fuzzy between kept title-similar entries without IDs
    # Build list of entries without DOI/PMID/arXiv
    no_id = [
        r
        for r in kept
        if not (
            _norm_doi(r.get("doi"))
            or _norm_pmid(r.get("pmid"))
            or _norm_arxiv(r.get("arxiv_id"))
        )
    ]
    used = set()
    final_kept = []
    # Simple O(n^2) since n is typically small (<200)
    for i, r in enumerate(no_id):
        if r["index"] in used:
            continue
        group = [r]
        used.add(r["index"])
        for j in range(i + 1, len(no_id)):
            s = no_id[j]
            if s["index"] in used:
                continue
            sim = _jaccard_trigram(r.get("title", ""), s.get("title", ""))
            if sim >= title_sim_threshold:
                group.append(s)
                used.add(s["index"])
        # choose primary
        base = group[0]
        for g in group[1:]:
            base = _choose_primary(base, g)
        final_kept.append(base)
        # update groups + merge_map
        gidx = []
        for g in group:
            merge_map[g["index"]] = base["index"]
            gidx.append(g["index"])
        groups.append(sorted(set(gidx)))

    # Add back those with IDs which were not in no_id list
    with_id = [r for r in kept if r not in no_id]
    all_kept = with_id + [r for r in final_kept if r not in with_id]

    # Deduplicate 'all_kept' by base index (some overlap may occur)
    unique_seen = set()
    unique_refs = []
    for r in all_kept:
        if r["index"] in unique_seen:
            continue
        unique_seen.add(r["index"])
        unique_refs.append(r)

    # Reassign contiguous 'index' for presentation? No: preserve original 'index' for mapping stability.
    return unique_refs, merge_map, groups


# ------------------------
# BM25 reranking (+ optional embedding cosine + domain boosts)
# ------------------------


def _idf(N: int, df: int) -> float:
    # BM25 idf variant; add small epsilon to avoid div by zero
    return math.log((N - df + 0.5) / (df + 0.5) + 1e-9)


def _bm25_scores(
    query: str, docs: List[str], k1: float = 1.5, b: float = 0.75
) -> List[float]:
    q_tokens = _tokenize(_fold(_clean_text(query)))
    if not q_tokens:
        return [0.0] * len(docs)
    doc_tokens = [_tokenize(_fold(_clean_text(d))) for d in docs]
    N = len(docs)
    avgdl = sum(len(t) for t in doc_tokens) / float(N or 1)
    # document frequencies
    df: Dict[str, int] = {}
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    scores = []
    for toks in doc_tokens:
        score = 0.0
        dl = len(toks) or 1
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in q_tokens:
            if t not in tf:
                continue
            idf = _idf(N, df.get(t, 0))
            denom = tf[t] + k1 * (1 - b + b * dl / (avgdl or 1))
            score += idf * (tf[t] * (k1 + 1)) / (denom or 1)
        scores.append(score)
    return scores


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rerank_refs(
    query: str,
    refs: List[dict],
    domain_terms: Optional[Set[str]] = None,
    embeddings: Optional[Dict[str, List[float]]] = None,
    query_embedding: Optional[List[float]] = None,
    top_k: int = 50,
    must_terms: Optional[Set[str]] = None,
    ban_terms: Optional[Set[str]] = None,
) -> List[dict]:
    """
    Combine BM25(title+abstract) with optional embedding cosine and domain-term boosts.
    - If 'embeddings' provided, key them by a stable ID per ref (e.g., DOI or fallback to str(index)).
    - If 'query_embedding' provided, cosine is added (weight 0.4).
    - Domain boost: +0.5 if >=2 domain terms present; +0.2 if 1 term.
    Returns top_k refs sorted by 'score' desc, with 'score' attached.
    """
    texts = [(r.get("title", "") + " " + r.get("abstract", "")).strip() for r in refs]
    bm25 = _bm25_scores(query, texts)
    # Precompute domain presence
    domain_terms = set(t.lower() for t in (domain_terms or set()))

    def domain_hits(txt: str) -> int:
        if not domain_terms or not txt:
            return 0
        T = set(_tokenize(_fold(_clean_text(txt))))
        return sum(1 for t in domain_terms if t in T)

    # Compute final
    out = []
    for i, r in enumerate(refs):
        score = 0.6 * bm25[i]
        # embedding cosine
        if embeddings and query_embedding:
            # pick a stable key
            key = (
                _norm_doi(r.get("doi"))
                or _norm_pmid(r.get("pmid"))
                or _norm_arxiv(r.get("arxiv_id"))
                or str(r.get("index"))
            )
            vec = embeddings.get(key)
            if vec:
                score += 0.4 * _cosine(query_embedding, vec)
        # domain boost
        hits = domain_hits(texts[i])
        if hits >= 2:
            score += 0.5
        elif hits == 1:
            score += 0.2
        # hard/soft topic constraints
        body_lc = (r.get("title", "") + " " + r.get("abstract", "")).lower()
        title_lc = (r.get("title", "") or "").lower()
        if must_terms:
            # If none of the must_terms appear anywhere, strongly penalize
            if not any(mt in body_lc for mt in must_terms):
                score -= 2.0
        if ban_terms:
            # If title contains any banned term, strongly penalize
            if any(bt in title_lc for bt in ban_terms):
                score -= 2.0
        # penalty if clearly off-topic (no shared tokens at all)
        if bm25[i] == 0.0 and hits == 0:
            score -= 0.25
        r2 = dict(r)
        r2["score"] = float(score)
        out.append(r2)

    out.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return out[:top_k]


# ------------------------
# High-level helpers
# ------------------------


def dedupe_and_rerank(
    query: str,
    refs: List[dict],
    domain_terms: Optional[Set[str]] = None,
    embeddings: Optional[Dict[str, List[float]]] = None,
    query_embedding: Optional[List[float]] = None,
    top_k: int = 50,
    must_terms: Optional[Set[str]] = None,
    ban_terms: Optional[Set[str]] = None,
) -> List[dict]:
    unique, merge_map, _ = dedupe_refs(refs)
    ranked = rerank_refs(
        query,
        unique,
        domain_terms=domain_terms,
        embeddings=embeddings,
        query_embedding=query_embedding,
        top_k=top_k,
        must_terms=must_terms,
        ban_terms=ban_terms,
    )
    return ranked


def split_used_refs(
    refs_all: List[dict], used_indexes: List[int]
) -> Tuple[List[dict], Dict[int, int]]:
    """
    Given full (ranked) refs with original 'index' values,
    return (refs_used_renumbered, index_map) where index_map maps old->new (1-based).
    """
    used_set = set(used_indexes or [])
    selected = [r for r in refs_all if r.get("index") in used_set]
    # Preserve order as in refs_all
    new_list = []
    index_map: Dict[int, int] = {}
    for i, r in enumerate(selected, 1):
        r2 = dict(r)
        old = r.get("index")
        r2["old_index"] = old
        r2["index"] = i  # renumbered for display
        index_map[old] = i
        new_list.append(r2)
    return new_list, index_map


def format_references_block(refs_used: List[dict]) -> str:
    """
    Render ACS-ish block from used refs (expected to be renumbered 1..m).
    Tries DOI first; falls back to URL.
    """
    lines = []
    for r in refs_used:
        idx = r.get("index", "?")
        authors = r.get("authors") or ""
        title = r.get("title") or ""
        venue = r.get("venue") or ""
        year = str(r.get("year") or "").strip()
        doi = _norm_doi(r.get("doi"))
        url = _strip(r.get("url") or "")
        link = f"https://doi.org/{doi}" if doi else url
        piece = f"[{idx}] {authors}. {title}. {venue} {year}. {link}".strip()
        # tidy spaces
        piece = re.sub(r"\\s+\\.", ".", piece)
        piece = re.sub(r"\\s{2,}", " ", piece)
        lines.append(piece)
    return "\\n".join(lines)


# A reasonable default domain-term set for nanochem / synthesis
DEFAULT_NANOCHEM_TERMS: Set[str] = set(
    map(
        str.lower,
        [
            "synthesis",
            "solvothermal",
            "hydrothermal",
            "autoclave",
            "nanocrystal",
            "nanoparticle",
            "nanowire",
            "nanorod",
            "nanocube",
            "seed",
            "precursor",
            "ligand",
            "surfactant",
            "oleylamine",
            "oleic",
            "PVP",
            "ethylene",
            "glycol",
            "polyol",
            "reduction",
            "nucleation",
            "growth",
            "facet",
            "{111}",
            "{100}",
            "HCl",
            "NaCl",
            "AgNO3",
            "Ag+",
            "AuCl3",
            "Fe(acac)3",
            "Ni(acac)2",
            "TOP",
            "TOPO",
            "HDA",
            "HDD",
            "OA",
            "reaction",
            "anneal",
            "calcination",
            "microwave",
            "stirring",
            "injection",
            "temperature",
            "time",
            "monodisperse",
            "monodispersity",
            "facet-selective",
            "shape-control",
            "capping",
            "cetyltrimethylammonium",
        ],
    )
)
