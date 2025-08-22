import httpx
from typing import List, Dict

ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"

def search_arxiv(query: str, max_results: int = 6, start: int = 0) -> List[Dict]:
    """
    Returns a list of entries with keys: id, title, summary, url, published, authors
    Raises only on network errors; 400s from arXiv are handled by sanitizing params.
    """
    try:
        mr = int(max_results)
    except Exception:
        mr = 6
    if mr <= 0: mr = 6
    try:
        st = int(start)
    except Exception:
        st = 0
    params = {"search_query": query or "", "start": st, "max_results": mr}
    r = httpx.get(ARXIV_ENDPOINT, params=params, timeout=30.0)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        # Try a safe fallback with tiny page size if arXiv rejects the request
        if r.status_code == 400:
            params["max_results"] = 5
            r = httpx.get(ARXIV_ENDPOINT, params=params, timeout=30.0)
            r.raise_for_status()
        else:
            raise

    text = r.text or ""
    # Parse with feedparser if available; otherwise naive parsing
    try:
        import feedparser  # type: ignore
        feed = feedparser.parse(text)
        out: List[Dict] = []
        for e in feed.entries:
            out.append({
                "id": getattr(e, "id", ""),
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "url": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
                "authors": [a.name for a in getattr(e, "authors", [])] if getattr(e, "authors", None) else [],
            })
        return out
    except Exception:
        # Naive fallback: extract <entry> blocks
        import re, html
        entries = re.findall(r"<entry>(.*?)</entry>", text, flags=re.S|re.I)
        out: List[Dict] = []
        def _tag(tag, s): 
            m = re.search(fr"<{tag}[^>]*>(.*?)</{tag}>", s, flags=re.S|re.I)
            return html.unescape(m.group(1).strip()) if m else ""
        for blk in entries:
            out.append({
                "id": _tag("id", blk),
                "title": _tag("title", blk),
                "summary": _tag("summary", blk),
                "url": _tag("link", blk) or _tag("id", blk),
                "published": _tag("published", blk),
                "authors": re.findall(r"<name>(.*?)</name>", blk, flags=re.S|re.I),
            })
        return out