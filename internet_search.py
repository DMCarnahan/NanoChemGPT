from __future__ import annotations

import os
import httpx
import functools
import json
import time
import urllib.parse
import re
from typing import Any, Dict, Iterable, List, Optional

BASE = "https://api.openalex.org/works"

# ---- User-Agent resolution (env-aware) ----
_DEFAULT_APP = "NanoChemGPT/1.0"
OPENALEX_BASE = "https://api.openalex.org/works"
OPENALEX_TEXT = "https://api.openalex.org/text"

_CONTACT_ENV_KEYS = ("OPENALEX_CONTACT", "CONTACT_EMAIL", "ADMIN_EMAIL")

_DEFAULT_APP = "NanoChemGPT/1.0"

def _get_contact_email() -> str:
    for k in _CONTACT_ENV_KEYS:
        v = os.getenv(k, "").strip()
        if "@" in v:
            return v
    return ""

def _build_user_agent_from_env() -> str:
    # 1) Respect explicit override
    ua = os.getenv("USER_AGENT", "").strip()
    if ua:
        return ua
    # 2) Otherwise include mailto if we have it
    email = _get_contact_email()
    return f"{_DEFAULT_APP} (mailto:{email})" if email else _DEFAULT_APP

USER_AGENT = _build_user_agent_from_env()
CONTACT_EMAIL = _get_contact_email()

HTTP_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

# ---- HTTP with retries ----
def _http_get_json(url: str, *, tries: int = 3, timeout: float = 20.0) -> dict | None:
    last_err = None
    for i in range(tries):
        try:
            with httpx.Client(headers=HTTP_HEADERS, timeout=timeout) as client:
                r = client.get(url)
            if r.status_code == 403:
                time.sleep(0.6 * (i + 1))
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (i + 1))
    print(f"[basic_search] OpenAlex fetch failed: {last_err}")
    return None

# --- Helpers -----------------

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
    # OpenAlex often returns a full URL; normalize to bare DOI
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi

def _pick_url(w: dict) -> str:
    pl = w.get("primary_location") or {}
    url = pl.get("landing_page_url") or (w.get("host_venue") or {}).get("url") or ""
    doi = _pick_doi(w)
    if (not url) and doi:
        url = f"https://doi.org/{doi}"
    return url

def _is_offtopic(rec: dict) -> bool:
    t = f"{rec.get('title','')} {rec.get('journal','')}".lower()
    bad = (
        "battery","supercapacitor","capacitor","fuel cell","electrode",
        "cathode","anode","voc","ceo2","ceria","adsorption","review"
    )
    return any(b in t for b in bad)

# --- Post-process OpenAlex results ------------------------------------------
def _postprocess_openalex_results(data: dict, n: int, query: str = "") -> list[dict]:
    """
    Convert OpenAlex 'works' payload into app records:
      {title, year, url, doi, abstract, authors, journal}
    Dedup, filter off-topic, and rank by simple relevance score.
    """
    works = (data or {}).get("results", []) or []
    out: list[dict] = []
    seen_titles = set()
    seen_dois = set()

    for w in works:
        title = (w.get("display_name") or "").strip()
        if not title:
            continue
        year = w.get("publication_year") or None
        doi = _pick_doi(w)
        url = _pick_url(w)
        abstract = _invert_openalex_abstract(w.get("abstract_inverted_index"))
        authors = [a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])]
        journal = (w.get("host_venue") or {}).get("display_name", "") or ""

        # Dedupe
        tkey = title.lower()
        if doi:
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
        else:
            if tkey in seen_titles:
                continue
            seen_titles.add(tkey)

        rec = {
            "title": title,
            "year": year,
            "url": url,
            "doi": doi or "",
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
        }
        if not _is_offtopic(rec):
            out.append(rec)

    # ---- Lightweight ranking ----------------------------------------------
    # Use your core terms for scoring + a tiny popularity nudge
    key_terms = set((_build_search_string(query, level="core") or "").lower().split())
    def _score(rec: dict) -> float:
        hay = f"{rec['title']} {rec['abstract']} {rec['journal']}".lower()
        base = sum(1.0 for k in key_terms if k and k in hay)
        # small bonus for “hot injection” (removed from URL to avoid WAF)
        if "hot injection" in hay or "hot-injection" in hay:
            base += 1.5
        # tiny popularity nudge if cited_by_count present
        # try to grab it from original works list by title match:
        try:
            w = next((_w for _w in works if (_w.get("display_name") or "").strip() == rec["title"]), None)
            c = (w or {}).get("cited_by_count") or 0
            base += (c ** 0.5) * 0.05
        except Exception:
            pass
        return base

    out.sort(key=_score, reverse=True)
    return out[:n]

_FORMULA_RX = re.compile(r"\b(?:[A-Z][a-z]?[\d]{0,3}){2,}\b")  # Cu2O, NiCo, Fe3O4, CoNi, etc.
# Common material-class words to catch 
_MAT_CLASSES = ["oxide","sulfide","selenide","telluride","nitride","phosphide",
                "carbide","boride","hydroxide","perovskite","spinel","alloy","intermetallic"]

# Lightweight element dictionary
_ELEMENT_WORDS = {
    "copper":"copper","cu":"copper",
    "nickel":"nickel","ni":"nickel",
    "cobalt":"cobalt","co":"cobalt",
    "iron":"iron","fe":"iron",
    "silver":"silver","ag":"silver",
    "gold":"gold","au":"gold",
    "zinc":"zinc","zn":"zinc",
    "tin":"tin","sn":"tin",
    "platinum":"platinum","pt":"platinum",
    "palladium":"palladium","pd":"palladium",
    "titanium":"titanium","ti":"titanium",
    "aluminum":"aluminum","aluminium":"aluminum","al":"aluminum",
}

# Broad morphology vocabulary
_MORPH = {
    "nanorod":["nanorod","nanorods","rod","rods"],
    "nanowire":["nanowire","nanowires","wire","wires"],
    "nanotube":["nanotube","nanotubes","tube","tubes"],
    "nanoribbon":["nanoribbon","nanoribbons","ribbon","ribbons"],
    "nanobelt":["nanobelt","nanobelts","belt","belts"],
    "nanoplate":["nanoplate","nanoplates","plate","plates","nanosheet","nanosheets","sheet","sheets","flake","flakes"],
    "nanocube":["nanocube","nanocubes","cube","cubes","octahedron","octahedra","sphere","spheres"],
}

_ROUTES = ["polyol","hydrothermal","solvothermal","electrodeposition","seed-mediated",
           "template","microwave","photochemical","galvanic","chemical reduction","PVP","CTAB",
           "oleylamine","ethylene glycol"]

_SCALE = ["facile","scalable","scaleable","gram-scale","large scale","high yield"]

def _tokenize_lower(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\-\+/\.]+", (s or "").lower())

def _extract_material_terms(question: str) -> List[str]:
    toks = _tokenize_lower(question)
    mats: List[str] = []

    # element names/symbols that appear
    for t in toks:
        if t in _ELEMENT_WORDS:
            mats.append(_ELEMENT_WORDS[t])

    # chemical formulas like Cu2O, NiCo, Fe3O4
    mats.extend(_FORMULA_RX.findall(question))

    # generic material classes present in text
    for cls in _MAT_CLASSES:
        if cls in toks:
            mats.append(cls)

    # de-dup, keep short
    mats = list(dict.fromkeys(mats))[:10]
    return mats

def _extract_morphology_terms(question: str) -> List[str]:
    q = (question or "").lower()
    found = []
    for group in _MORPH.values():
        if any(w in q for w in group):
            found.extend(group)
    if not found:
        for key in ("nanorod","nanowire","nanotube","nanoribbon","nanobelt"):
            found.extend(_MORPH[key])
    return list(dict.fromkeys(found))

def _or_group(words: List[str]) -> str:
    words = [w for w in dict.fromkeys(words) if w]  # de-dup, keep order
    return " OR ".join(f'"{w}"' if " " in w else w for w in words)

def _build_boolean_query(question: str) -> str:
    mats  = _extract_material_terms(question)        # optional
    morph = _extract_morphology_terms(question)      # required/fallback
    routes = _ROUTES
    scale  = _SCALE

    clauses = []
    if mats:   clauses.append(f"({_or_group(mats)})")
    if morph:  clauses.append(f"({_or_group(morph)})")
    if routes: clauses.append(f"({_or_group(routes)})")
    if scale:  clauses.append(f"({_or_group(scale)})")
    return " AND ".join(clauses)

def _strip_openalex_id(x: str) -> str:
    # accepts 'https://openalex.org/T123' or 'T123' -> 'T123'
    return x.rsplit("/", 1)[-1]

def _aboutness_topics_from_text(text: str, k: int = 3) -> list[str]:
    url = f"{OPENALEX_TEXT}?{urllib.parse.urlencode({'title': text[:2000], 'mailto': CONTACT_EMAIL})}"
    data = _http_get_json(url) or {}
    out = []
    pt = (data.get("primary_topic") or {}).get("id")
    if pt:
        out.append(_strip_openalex_id(pt))
    for t in (data.get("topics") or [])[:k]:
        tid = t.get("id")
        if tid:
            tid = _strip_openalex_id(tid)
            if tid not in out:
                out.append(tid)
    return out

def _is_offtopic(rec: dict) -> bool:
    t = (rec.get("title","") + " " + rec.get("journal","")).lower()
    bad = ("battery","supercapacitor","review")
    return any(b in t for b in bad)

WAF_BLOCK_TERMS = {"injection", "sql", "payload", "drop", "truncate"}  # keep small; 'injection' is the big one

def _build_search_string(question: str, *, level: str = "full") -> str:
    """
    Build a plain string for OpenAlex `search=`.
    level: 'full' (material+morph+routes+scale),
           'core' (material+morph+scale),
           'minimal' (morph only).
    """
    mats  = _extract_material_terms(question)
    morph = _extract_morphology_terms(question)
    routes = _ROUTES
    scale  = _SCALE

    terms = []
    # always include morphology
    terms.extend(morph)
    # include material if present
    terms.extend(mats)

    if level == "full":
        terms.extend(routes)
        terms.extend(scale)
    elif level == "core":
        terms.extend(scale)
    # 'minimal' adds nothing more

    # sanitize: split tokens by spaces/hyphens and drop WAF triggers
    safe_terms = []
    for t in dict.fromkeys(terms):
        if not t:
            continue
        pieces = t.replace("–", "-").replace("/", " ").split()
        if any(p.lower() in WAF_BLOCK_TERMS for p in pieces):
            continue
        safe_terms.append(t)

    return " ".join(safe_terms)

# ---- Query builder ----
def _build_url_smart(question: str, *, n: int = 6, from_year: int | None = 2005,
                     lang: str = "en", use_aboutness: bool = True) -> tuple[str, list[str]]:
    per_page = max(1, min(n, 25))
    filters: List[str] = []

    # Narrow by type/year/language ONLY via filters
    filters.append("type:journal-article")
    if lang:
        filters.append(f"language:{lang}")
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")

    tids: List[str] = []
    if use_aboutness:
        tids = _aboutness_topics_from_text(question, k=3)
        if tids:
            filters.append("topics.id:" + "|".join(tids))  # let urlencode encode pipes

    params = {
        "per-page": str(per_page),
        "sort": "relevance_score:desc",  # works with `search=`
        "select": ",".join([
            "id","display_name","publication_year","primary_location",
            "doi","abstract_inverted_index","authorships","host_venue",
            "type","language","topics","primary_topic","cited_by_count"
        ]),
        "filter": ",".join(filters),
        "search": _build_search_string(question),
    }
    if CONTACT_EMAIL:
        params["mailto"] = CONTACT_EMAIL

    qs = urllib.parse.urlencode(params, doseq=True, safe=':," ')  # no parentheses or '|'
    return f"{BASE}?{qs}", tids

def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return s

def _pick_url(work: dict) -> str:
    # Prefer landing page, else OpenAlex id
    loc = work.get("primary_location") or {}
    return _norm_str(loc.get("landing_page_url") or work.get("id", ""))

def _pick_doi(work: dict) -> str:
    doi = work.get("doi") or ""
    doi = _norm_str(doi)
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "", 1)
    return doi

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

    # lightweight on-topic filter using your keywords
    key = set(_build_search_string(query, level="core").lower().split())
    scored = []
    for r in out:
        text = f"{r['title']} {r['abstract']} {r['journal']}".lower()
        score = sum(1 for k in key if k in text)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:n]]

def _is_403(err: Exception) -> bool:
    return "403" in str(err)

def search_papers(query: str, n: int = 6, *,
                  from_year: Optional[int] = 2005,
                  use_aboutness: bool = True) -> List[dict]:
    q = (query or "").strip()
    if not q:
        return []

    # try full search (routes+scale)
    for level in ("full", "core", "minimal"):
        url, tids = _build_url_smart(q, n=n, from_year=from_year, use_aboutness=use_aboutness, level=level)
        try:
            data = _http_get_json(url)
        except Exception as e:
            data = None

        if data and data.get("results"):
            return _postprocess_openalex_results(data, n)

        # drop topics if we had them
        if (not data) and tids:
            url, _ = _build_url_smart(q, n=n, from_year=from_year, use_aboutness=False, level=level)
            data = _http_get_json(url)
            if data and data.get("results"):
                return _postprocess_openalex_results(data, n)

    # FINAL FALLBACK: Crossref
    x = _crossref_search(q, n=n, from_year=from_year)
    if x:
        return x

    return []

def set_user_agent(email: str) -> None:
    """
    Set a compliant User-Agent and contact email.
    """
    global USER_AGENT, CONTACT_EMAIL, HTTP_HEADERS
    email = (email or "").strip()
    if not ("@" in email):
        raise ValueError("Please provide a valid email address for the User-Agent.")
    CONTACT_EMAIL = email
    USER_AGENT = f"NanoChemGPT/1.0 (mailto:{email})"
    HTTP_HEADERS = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

def build_chem_query(chem: str, property: str = "", method: str = "", extra: str = "") -> str:
    parts = [chem]
    if property: parts.append(property)
    if method: parts.append(method)
    if extra: parts.append(extra)
    return " ".join(parts)

def filter_results(results: List[dict], keywords: List[str]) -> List[dict]:
    filtered = []
    for r in results:
        text = (r.get("title", "") + " " + r.get("abstract", "")).lower()
        if all(k.lower() in text for k in keywords):
            filtered.append(r)
    return filtered if filtered else results  # fallback to all if none match

__all__ = ["search_papers", "set_user_agent"]

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Search OpenAlex for works.")
    ap.add_argument("query", nargs="*", help="search terms")
    ap.add_argument("-n", type=int, default=6, help="max results (default 6)")
    ap.add_argument("--from-year", type=int, default=None)
    ap.add_argument("--to-year", type=int, default=None)
    ap.add_argument("--is-oa", choices=["true","false"], default=None)
    ap.add_argument("--sort", default="cited_by_count:desc", help="OpenAlex sort (default cited_by_count:desc)")
    ap.add_argument("--email", default=None, help="contact email for User-Agent")
    args = ap.parse_args()

    if args.email:
        try:
            set_user_agent(args.email)
        except Exception as e:
            print(f"warning: {e}", file=sys.stderr)

    is_oa = None
    if args.is_oa == "true":
        is_oa = True
    elif args.is_oa == "false":
        is_oa = False

    query = " ".join(args.query).strip()
    if not query:
        ap.print_help()
        sys.exit(2)

    try:
        items = search_papers(
            query,
            n=args.n,
            from_year=args.from_year,
            use_aboutness=True
        )
        print(json.dumps(items, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
