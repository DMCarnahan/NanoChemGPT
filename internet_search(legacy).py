
from __future__ import annotations

import os
import time
import httpx
import urllib.parse
import re
from typing import List, Dict, Optional, Tuple

BASE = "https://api.openalex.org/works"
TEXT = "https://api.openalex.org/text"

# ---- Contact / headers ------------------------------------------------------
_DEFAULT_APP = "NanoChemGPT/1.0"
_CONTACT_ENV_KEYS = ("OPENALEX_CONTACT", "CONTACT_EMAIL", "ADMIN_EMAIL")

def _get_contact_email() -> str:
    for k in _CONTACT_ENV_KEYS:
        v = os.getenv(k, "").strip()
        if "@" in v:
            return v
    return ""

def _build_user_agent_from_env() -> str:
    ua = os.getenv("USER_AGENT", "").strip()
    if ua:
        return ua
    email = _get_contact_email()
    return f"{_DEFAULT_APP} (mailto:{email})" if email else _DEFAULT_APP

USER_AGENT = _build_user_agent_from_env()
CONTACT_EMAIL = _get_contact_email()
HTTP_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

# ---- HTTP helper with retries ----------------------------------------------
def _http_get_json(url: str, *, tries: int = 3, timeout: float = 20.0) -> dict | None:
    last_err = None
    for i in range(tries):
        try:
            with httpx.Client(headers=HTTP_HEADERS, timeout=timeout) as client:
                r = client.get(url)
            if r.status_code in (429, 403):
                time.sleep(0.6 * (i + 1))
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (i + 1))
    print(f"[internet_search] fetch failed: {last_err}")
    return None

# ---- Gating for OpenAlex /text ---------------------------------------------
def _should_use_aboutness(q: str) -> bool:
    if not q:
        return False
    q = q.strip()
    if len(q) > 80:
        return False
    if any(ch in q for ch in ",;:!?/\|"):
        return False
    return True

# ---- Post-processing --------------------------------------------------------
def _invert_openalex_abstract(inv: dict | None) -> str:
    if not isinstance(inv, dict) or not inv:
        return ""
    size = max(p for positions in inv.values() for p in positions) + 1
    words = [""] * size
    for token, positions in inv.items():
        for p in positions:
            if 0 <= p < size:
                words[p] = token
    return " ".join(w for w in words if w)

def _pick_doi(w: dict) -> str:
    doi = (w.get("doi") or "").strip()
    for pref in ("https://doi.org/", "http://doi.org/"):
        if doi.lower().startswith(pref):
            return doi[len(pref):]
    return doi

def _pick_url(w: dict) -> str:
    pl = w.get("primary_location") or {}
    boa = w.get("best_oa_location") or {}
    url = pl.get("landing_page_url") or boa.get("landing_page_url")
    if not url:
        for loc in (w.get("locations") or []):
            url = loc.get("landing_page_url")
            if url:
                break
    if not url:
        doi = _pick_doi(w)
        if doi:
            url = f"https://doi.org/{doi}"
    return url or ""

def _pick_journal_name(w: dict) -> str:
    pl = w.get("primary_location") or {}
    src = pl.get("source") or {}
    name = (src.get("display_name") or "").strip()
    if name:
        return name
    for loc in (w.get("locations") or []):
        s = (loc.get("source") or {}).get("display_name")
        if s:
            return (s or "").strip()
    return ""

def _postprocess_openalex_results(data: dict, n: int, query: str = "") -> list[dict]:
    works = (data or {}).get("results", []) or []
    out: list[dict] = []
    seen_titles = set()
    seen_dois = set()

    for w in works:
        title = (w.get("title") or w.get("display_name") or "").strip()
        if not title:
            continue
        year = w.get("publication_year") or None
        doi = _pick_doi(w)
        url = _pick_url(w)
        abstract = _invert_openalex_abstract(w.get("abstract_inverted_index"))
        authors = [a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])]
        journal = _pick_journal_name(w)

        tkey = title.lower()
        if doi:
            if doi in seen_dois: continue
            seen_dois.add(doi)
        else:
            if tkey in seen_titles: continue
            seen_titles.add(tkey)

        out.append({
            "title": title,
            "year": year,
            "url": url,
            "doi": doi,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
        })

    # Simple query-aware sorting to nudge relevance
    key_terms = set((_build_search_string(query, level="core") or "").lower().split())

    def _score(rec: dict) -> float:
        hay = f"{rec.get('title','')} {rec.get('abstract','')} {rec.get('journal','')}".lower()
        return sum(1.0 for k in key_terms if k and k in hay)

    out.sort(key=_score, reverse=True)
    return out[:n]

# ---- Query construction -----------------------------------------------------
_FORMULA_RX = re.compile(r"\b(?:[A-Z][a-z]?\d*){1,5}\b")
_MAT_CLASSES = ["oxide","sulfide","selenide","telluride","nitride","phosphide",
                "carbide","boride","hydroxide","perovskite","spinel","alloy","intermetallic"]

_MORPH = {
    "nanorod":["nanorod","nanorods","rod","rods"],
    "nanowire":["nanowire","nanowires","wire","wires"],
    "nanotube":["nanotube","nanotubes","tube","tubes"],
    "nanoribbon":["nanoribbon","nanoribbons","ribbon","ribbons"],
    "nanobelt":["nanobelt","nanobelts","belt","belts"],
    "nanoplate":["nanoplate","nanoplates","plate","plates","nanosheet","nanosheets","sheet","sheets","flake","flakes"],
    "nanocube":["nanocube","nanocubes","cube","cubes","octahedron","octahedra","sphere","spheres"],
}

_ROUTES = [
    "polyol","hydrothermal","solvothermal","autoclave",
    "electrodeposition","seed-mediated","template","microwave",
    "photochemical","galvanic","chemical reduction","pvp","ctab",
    "oleylamine","ethylene glycol",
]

_SCALE = ["facile","scalable","scaleable","gram-scale","large scale","high yield"]

def _tokenize_lower(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\-\+/\.]+", (s or "").lower())

def _extract_material_terms(question: str) -> List[str]:
    toks = _tokenize_lower(question)
    mats: List[str] = []
    # element words (very light)
    E = {
        "mn":"manganese","ni":"nickel","co":"cobalt","fe":"iron","cu":"copper","zn":"zinc",
        "ti":"titanium","al":"aluminum","si":"silicon","pb":"lead","ag":"silver","au":"gold"
    }
    for t in toks:
        if t in E:
            mats.append(E[t])
        elif t in E.values():
            mats.append(t)
    # chemical formulas
    mats.extend(_FORMULA_RX.findall(question))
    # class nouns
    for cls in _MAT_CLASSES:
        if cls in toks:
            mats.append(cls)
    return list(dict.fromkeys(mats))[:10]

def _extract_morphology_terms(question: str) -> List[str]:
    q = (question or "").lower()
    found = []
    for group in _MORPH.values():
        if any(w in q for w in group):
            found.extend(group)
    return list(dict.fromkeys(found))

def _or_group(words: List[str]) -> str:
    words = [w for w in dict.fromkeys(words) if w]
    return " OR ".join(f'"{w}"' if " " in w else w for w in words)

def _build_boolean_query(question: str) -> str:
    mats  = _extract_material_terms(question)
    morph = _extract_morphology_terms(question)
    routes = _ROUTES
    scale  = _SCALE

    clauses = []
    if mats:   clauses.append(f"({_or_group(mats)})")
    if morph:  clauses.append(f"({_or_group(morph)})")
    if routes: clauses.append(f"({_or_group(routes)})")
    if scale:  clauses.append(f"({_or_group(scale)})")
    return " AND ".join(clauses)

WAF_BLOCK_TERMS = {"injection", "sql", "payload", "drop", "truncate"}

def _build_search_string(question: str, *, level: str = "full") -> str:
    mats  = _extract_material_terms(question)
    morph = _extract_morphology_terms(question)
    routes = _ROUTES
    scale  = _SCALE

    terms = []
    terms.extend(morph)
    terms.extend(mats)
    if level == "full":
        terms.extend(routes); terms.extend(scale)
    elif level == "core":
        terms.extend(scale)

    safe_terms = []
    for t in dict.fromkeys(terms):
        if not t:
            continue
        pieces = t.replace("–", "-").replace("/", " ").split()
        if any(p.lower() in WAF_BLOCK_TERMS for p in pieces):
            continue
        safe_terms.append(t)

    return " ".join(safe_terms)

# ---- URL builder ------------------------------------------------------------
def _build_url_smart(question: str, *,
                     n: int = 6,
                     from_year: Optional[int] = 2005,
                     lang: str = "en",
                     use_aboutness: bool = True,
                     level: str = "full",
                     topics: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    per_page = max(10, min(50, n))
    filters = ["type:journal-article"]
    if lang:
        filters.append(f"language:{lang}")
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")

    # topics are provided by the caller; we never call /text here
    tids = list(topics or [])
    if use_aboutness and tids:
        filters.append("topics.id:" + "|".join(tids))

    params = {
        "per_page": per_page,
        "sort": "cited_by_count:desc",
        "search": _build_search_string(question, level=level),
        "filter": ",".join(filters),
    }
    if CONTACT_EMAIL:
        params["mailto"] = CONTACT_EMAIL

    qs = urllib.parse.urlencode(params, doseq=True, safe=':," ')
    return f"{BASE}?{qs}", tids

# ---- Crossref fallback ------------------------------------------------------
def _crossref_search(query: str, n: int = 6, from_year: Optional[int] = 2005) -> List[dict]:
    base = "https://api.crossref.org/works"
    params = {
        "query": _build_search_string(query, level="core"),
        "rows": str(n * 2),
        "filter": ",".join([f"from-pub-date:{from_year}-01-01"] if from_year else []),
        "select": "title,DOI,URL,issued,container-title,type,abstract,author"
    }
    if CONTACT_EMAIL:
        params["mailto"] = CONTACT_EMAIL
    url = f"{base}?{urllib.parse.urlencode(params)}"
    data = _http_get_json(url) or {}
    items = ((data.get("message") or {}).get("items") or [])[:n*2]

    out = []
    for it in items:
        title = " ".join(it.get("title") or []) or ""
        year = None
        try:
            year = (it.get("issued") or {}).get("date-parts", [[None]])[0][0]
        except Exception:
            pass
        out.append({
            "title": title,
            "year": year,
            "url": it.get("URL") or (f"https://doi.org/{it.get('DOI')}" if it.get("DOI") else ""),
            "doi": it.get("DOI") or "",
            "abstract": (it.get("abstract") or "").replace("\n", " ").strip(),
            "authors": [a.get("family","") + (", " + a.get("given","") if a.get("given") else "") for a in it.get("author", []) if isinstance(a, dict)],
            "journal": (it.get("container-title") or [""])[0],
        })

    key = set(_build_search_string(query, level="core").lower().split())
    scored = []
    for r in out:
        text = f"{r['title']} {r['abstract']} {r['journal']}".lower()
        score = sum(1 for k in key if k in text)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:n]]

# ---- Public API -------------------------------------------------------------
def search_papers(q: str, n: int = 20, from_year: int = 2005, use_aboutness: bool = True) -> list[dict]:
    q = (q or "").strip()
    use_aboutness = bool(use_aboutness) and _should_use_aboutness(q)

    # compute topics once (optional)
    tids: list[str] = []
    if use_aboutness:
        try:
            u = f"{TEXT}?{urllib.parse.urlencode({'text': q[:2000], 'mailto': CONTACT_EMAIL})}"
            data = _http_get_json(u) or {}
            pt = (data.get("primary_topic") or {}).get("id")
            if pt:
                tids.append(pt.rsplit("/", 1)[-1])
            for t in (data.get("topics") or [])[:3]:
                tid = (t.get("id") or "").rsplit("/", 1)[-1]
                if tid and tid not in tids:
                    tids.append(tid)
        except Exception as e:
            print("[internet_search] aboutness disabled due to error:", e)
            use_aboutness = False
            tids = []

    for level in ("full", "core", "minimal"):
        url, _ = _build_url_smart(q, n=n, from_year=from_year, use_aboutness=use_aboutness, level=level, topics=tids)
        data = _http_get_json(url)
        if data and data.get("results"):
            out = _postprocess_openalex_results(data, n, query=q)
            if out:
                return out

        # If topics were applied but yielded nothing, retry once without topics
        if use_aboutness and tids:
            url, _ = _build_url_smart(q, n=n, from_year=from_year, use_aboutness=False, level=level, topics=[])
            data = _http_get_json(url)
            if data and data.get("results"):
                out = _postprocess_openalex_results(data, n, query=q)
                if out:
                    return out

    # Crossref fallback
    x = _crossref_search(q, n=n, from_year=from_year)
    return x or []

def set_user_agent(email: str) -> None:
    global USER_AGENT, CONTACT_EMAIL, HTTP_HEADERS
    email = (email or "").strip()
    if "@" not in email:
        raise ValueError("Please provide a valid email for the User-Agent.")
    CONTACT_EMAIL = email
    USER_AGENT = f"{_DEFAULT_APP} (mailto:{email})"
    HTTP_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

__all__ = ["search_papers", "set_user_agent"]
