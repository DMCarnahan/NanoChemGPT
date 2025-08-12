from __future__ import annotations

import os
from warnings import filters
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
_CONTACT_ENV_KEYS = ("OPENALEX_CONTACT_EMAIL", "CONTACT_EMAIL", "ADMIN_EMAIL")

def _build_user_agent_from_env() -> Optional[str]:
    ua = os.getenv("USER_AGENT", "").strip()
    if ua:
        return ua
    for key in _CONTACT_ENV_KEYS:
        email = os.getenv(key, "").strip()
        if email and "@" in email:
            return f"{_DEFAULT_APP} (+mailto:{email})"
    return None


USER_AGENT: str = _build_user_agent_from_env() or _DEFAULT_APP

# ---- TTL cache (5 minutes) ----
_TTL_SECONDS = 300
_cache: Dict[str, tuple[float, Any]] = {}  # url -> (timestamp, payload)

def _get_cached(url: str) -> Optional[Any]:
    ts_payload = _cache.get(url)
    if not ts_payload:
        return None
    ts, payload = ts_payload
    if (time.time() - ts) > _TTL_SECONDS:
        _cache.pop(url, None)
        return None
    return payload

def _set_cached(url: str, payload: Any) -> None:
    _cache[url] = (time.time(), payload)

# ---- HTTP with retries ----
def _http_get_json(url: str, *, timeout: float = 30.0, retries: int = 3, backoff: float = 0.75) -> Any:
    cached = _get_cached(url)
    if cached is not None:
        return cached

    last_exc: Optional[Exception] = None
    headers = {"User-Agent": USER_AGENT or _DEFAULT_APP}
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=timeout, headers=headers) as client:
                resp = client.get(url)
                resp.raise_for_status()
                # Try JSON; if it fails, raise
                data = resp.json()
                _set_cached(url, data)
                return data
        except Exception as e:
            last_exc = e
            if attempt >= retries:
                break
            time.sleep(backoff * attempt)  # simple linear backoff
    raise RuntimeError(f"GET failed after {retries} attempts: {last_exc}")


def _invert_openalex_abstract(inv: Any) -> str:
    """Rebuild plain-text abstract from OpenAlex abstract_inverted_index."""
    if not isinstance(inv, dict) or not inv:
        return ""
    size = max(p for positions in inv.values() for p in positions) + 1
    words = [""] * size
    for token, positions in inv.items():
        for p in positions:
            if 0 <= p < size:
                words[p] = token
    return " ".join(w for w in words if w)

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

_ROUTES = ["polyol","hydrothermal","solvothermal","electrodeposition","seed-mediated","hot injection",
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

def _aboutness_topics_from_text(text: str, k: int = 3) -> list[str]:
    # Use OpenAlex /text endpoint to get Topics, then return top-k IDs
    base = "https://api.openalex.org/text"
    url = f"{base}?{urllib.parse.urlencode({'title': text[:2000]})}"
    data = _http_get_json(url) or {}
    topic_ids = []
    pt = (data.get("primary_topic") or {}).get("id")
    if pt:
        topic_ids.append(pt)
    for t in (data.get("topics") or [])[:k]:
        tid = t.get("id")
        if tid and tid not in topic_ids:
            topic_ids.append(tid)
    return topic_ids

def _is_offtopic(rec: dict) -> bool:
    t = (rec.get("title","") + " " + rec.get("journal","")).lower()
    bad = ("battery","supercapacitor","review")
    return any(b in t for b in bad)

# ---- Query builder ----

def _build_url_smart(question: str, *, n: int = 6, from_year: int | None = 2005,
                     lang: str = "en", use_aboutness: bool = True) -> str:
    per_page = max(1, min(n, 25))
    filters = []

    # 1) Title+abstract boolean search (precise)
    q = _build_boolean_query(question)
    filters.append(f'title_and_abstract.search:({q})')

    # 2) Narrow by type/year/language
    filters.append("type:journal-article")
    if lang:
        filters.append(f"language:{lang}")
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")

    # 3) Optional topic filter using /text aboutness
    if use_aboutness:
        tids = _aboutness_topics_from_text(question, k=3)
        if tids:
            filters.append("topics.id:" + "|".join(tids))

    params = {
        # use hyphenated key (official)
        "per-page": str(per_page),
        # relevance_score is available because we used a search filter
        "sort": "relevance_score:desc",
        # keep response lean
        "select": ",".join([
            "id","display_name","publication_year","primary_location",
            "doi","abstract_inverted_index","authorships","host_venue",
            "type","language","topics","primary_topic","cited_by_count"
        ])
    }
    params["filter"] = ",".join(filters)
    qs = urllib.parse.urlencode(params, doseq=True, safe=":|(),\" ")
    return f"{BASE}?{qs}"

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

def search_papers(query: str, n: int = 6, *,
                  from_year: Optional[int] = 2005,
                  use_aboutness: bool = True) -> List[dict]:
    """
    Search tuned for materials synthesis questions.
    Returns dicts with title, year, url, doi, abstract, authors, journal.
    """
    q = (query or "").strip()
    if not q:
        return []

    url = _build_url_smart(q, n=n, from_year=from_year, use_aboutness=use_aboutness)
    data = _http_get_json(url) or {}
    results = data.get("results", []) or []

    out: List[dict] = []
    for w in results[:n*2]:  
        abs_text = _invert_openalex_abstract(w.get("abstract_inverted_index"))
        rec = {
            "title": _norm_str(w.get("display_name")),
            "year":  w.get("publication_year") or "",
            "url":   _pick_url(w),
            "doi":   _pick_doi(w),
            "abstract": abs_text,
            "authors": [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])],
            "journal": (w.get("host_venue") or {}).get("display_name", ""),
        }
        if not _is_offtopic(rec):
            out.append(rec)

    return out[:n] or out[:n]  # fall back gracefully

def set_user_agent(email: str) -> None:
    """
    Set a compliant User-Agent with a contact email, per OpenAlex guidelines.
    """
    global USER_AGENT
    email = (email or "").strip()
    if email and "@" in email:
        USER_AGENT = f"NanoChemGPT/1.0 (+mailto:{email})"
    else:
        raise ValueError("Please provide a valid email address for the User-Agent.")

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
