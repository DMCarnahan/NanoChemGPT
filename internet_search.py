"""
Lightweight helper around the public OpenAlex REST API.
Rate limit: 30 req / minute / IP (official docs).
"""
from __future__ import annotations
import httpx, functools, time, urllib.parse

BASE = "https://api.openalex.org/works"
UA   = {"User-Agent": "NanoChemGPT/1.0 (mailto:you@example.com)"}

@functools.lru_cache(maxsize=2048)
def _cached_get(url: str) -> dict:
    """Tiny TTL cache (5 min) to stay polite."""
    resp = httpx.get(url, headers=UA, timeout=30)
    resp.raise_for_status()
    return resp.json()

def search_papers(query: str, n: int = 6) -> list[dict]:
    """
    Return up to *n* dicts with at least title, year, url, doi – the
    same keys `/ask` already expects.
    """
    if not query.strip():
        return []
    url = (
        f"{BASE}?search={urllib.parse.quote_plus(query)}"
        f"&per_page={n}&sort=cited_by_count:desc"
    )
    data = _cached_get(url).get("results", [])
    out = []
    for w in data:
        loc   = w.get("primary_location", {})
        ident = w.get("id", "")
        out.append({
            "title": w.get("display_name", ""),
            "year":  w.get("publication_year", ""),
            "url":   loc.get("landing_page_url") or ident,
            "doi":   (w.get("doi") or "").replace("https://doi.org/", ""),
        })
    return out
