import httpx
from typing import List, Dict, Any

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

def _to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def search_eupmc(query: str, since_year=None, max_results=25) -> List[Dict[str, Any]]:
    """
    Return Europe PMC search results as a list of dicts.

    Args:
        query: free-text query string (Europe PMC syntax supported)
        since_year: filter to pubYear >= since_year (str or int accepted)
        max_results: cap on number of items to return (str or int accepted)

    The function is defensive:
      - Casts inputs to ints, clamps page size.
      - Uses 'params' (no manual URL concat).
      - On HTTP 400, retries with a smaller page size.
    """
    mr = _to_int(max_results, 25)
    if mr is None or mr <= 0:
        mr = 25
    # Europe PMC allows large pages but keep it reasonable
    page_size = min(mr, 1000)

    sy = _to_int(since_year, None)

    params = {
        "query": query or "",
        "format": "json",
        "pageSize": page_size,
    }

    try:
        r = httpx.get(EPMC_SEARCH, params=params, timeout=45.0)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Retry with tiny page if server rejected the request
            if r.status_code == 400:
                params["pageSize"] = 25
                r = httpx.get(EPMC_SEARCH, params=params, timeout=45.0)
                r.raise_for_status()
            else:
                raise
        js = r.json()
    except Exception as e:
        # Surface a minimal, traceable error structure in-band if caller wants to log it
        return []

    results = js.get("resultList", {}).get("result", []) or []
    out: List[Dict[str, Any]] = []

    for rec in results:
        try:
            year = _to_int(rec.get("pubYear"), None)
            if sy is not None and (year is None or year < sy):
                continue
            out.append({
                "id": rec.get("id") or rec.get("pmid") or rec.get("pmcid") or "",
                "title": rec.get("title") or "",
                "url": rec.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url", "") if rec.get("fullTextUrlList") else "",
                "doi": rec.get("doi") or "",
                "journal": rec.get("journalTitle") or "",
                "pubYear": year,
                "authors": rec.get("authorString") or "",
                "source": rec.get("source") or "",
                "isOpenAccess": rec.get("isOpenAccess"),
            })
        except Exception:
            continue

    # Cap to mr
    return out[:mr]
