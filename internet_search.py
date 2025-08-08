from __future__ import annotations

import httpx
import functools
import json
import time
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional

BASE = "https://api.openalex.org/works"

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
    headers = {"User-Agent": USER_AGENT}
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

# ---- Query builder ----
def _build_url(query: str, *, n: int = 6, sort: str = "cited_by_count:desc",
               from_year: Optional[int] = None, to_year: Optional[int] = None,
               is_oa: Optional[bool] = None) -> str:
    if n <= 0:
        n = 1
    per_page = max(1, min(n, 25))  # OpenAlex caps per_page at 200; 25 keeps payloads small
    params = {
        "search": query,
        "per_page": str(per_page),
        "sort": sort,
    }
    # Build filters
    filters = []
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    if to_year:
        filters.append(f"to_publication_date:{to_year}-12-31")
    if is_oa is True:
        filters.append("is_oa:true")
    if is_oa is False:
        filters.append("is_oa:false")
    if filters:
        params["filter"] = ",".join(filters)

    qs = urllib.parse.urlencode(params, doseq=True, safe=":,")
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

def search_papers(query: str, n: int = 6, *, sort: str = "cited_by_count:desc",
                  from_year: Optional[int] = None, to_year: Optional[int] = None,
                  is_oa: Optional[bool] = None) -> List[dict]:
    """
    Return up to *n* dicts with keys: title, year, url, doi.
    Optional filters: sort, from_year, to_year, is_oa.
    """
    q = (query or "").strip()
    if not q:
        return []

    url = _build_url(q, n=n, sort=sort, from_year=from_year, to_year=to_year, is_oa=is_oa)
    data = _http_get_json(url) or {}
    results = data.get("results", []) or []

    out: List[dict] = []
    for w in results[:n]:
        out.append({
            "title": _norm_str(w.get("display_name")),
            "year":  w.get("publication_year") or "",
            "url":   _pick_url(w),
            "doi":   _pick_doi(w),
        })
    return out

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
        items = search_papers(query, n=args.n, sort=args.sort,
                              from_year=args.from_year, to_year=args.to_year, is_oa=is_oa)
        print(json.dumps(items, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
