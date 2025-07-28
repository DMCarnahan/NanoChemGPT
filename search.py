import os, re, json, requests
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET
from typing import List, Dict, Optional

def _norm(title: Optional[str]) -> str:
    return (title or "").strip()

def _mk_ref(title=None, url=None, venue=None, year=None, doi=None, source=None):
    # Prefer landing page; if missing but DOI exists, build doi.org link
    if not url and doi:
        url = f"https://doi.org/{doi}"
    return {
        "title": _norm(title),
        "url": url,
        "venue": (venue or None),
        "year": (int(year) if str(year).isdigit() else year),
        "doi": (doi or None),
        "source": source or "unknown",
    }

def _dedupe(items: List[Dict], limit: int) -> List[Dict]:
    seen, out = set(), []
    for r in items:
        key = None
        if r.get("doi"):
            key = f"doi:{r['doi'].lower()}"
        elif r.get("url"):
            key = f"url:{r['url'].lower()}"
        elif r.get("title"):
            key = f"title:{r['title'].lower()}"
        if key and key not in seen and r.get("title"):
            seen.add(key); out.append(r)
        if len(out) >= limit:
            break
    return out

# ---------- Open providers (no key required) ----------
def _openalex(query: str, n: int) -> List[Dict]:
    url = (
        "https://api.openalex.org/works?"
        f"search={quote_plus(query)}&per_page={n}"
        "&select=display_name,doi,publication_year,host_venue,primary_location"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    out = []
    for w in data.get("results", []):
        title = w.get("display_name")
        pl = (w.get("primary_location") or {})
        url = pl.get("landing_page_url") or pl.get("pdf_url")
        venue = (w.get("host_venue") or {}).get("display_name")
        year = w.get("publication_year")
        doi  = w.get("doi")
        out.append(_mk_ref(title, url, venue, year, doi, "OpenAlex"))
    return out

def _crossref(query: str, n: int) -> List[Dict]:
    url = f"https://api.crossref.org/works?query={quote_plus(query)}&rows={n}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
    except Exception:
        return []
    out = []
    for it in items:
        title = " ".join(it.get("title", []) or [])
        link  = it.get("URL")
        cont  = (it.get("container-title") or [""])[0]
        year  = ((it.get("issued", {}).get("date-parts") or [[None]])[0] or [None])[0]
        doi   = it.get("DOI")
        out.append(_mk_ref(title, link, cont, year, doi, "Crossref"))
    return out

def _arxiv(query: str, n: int) -> List[Dict]:
    url = f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}&start=0&max_results={n}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("a:entry", ns):
        title = entry.findtext("a:title", default="", namespaces=ns).strip()
        link  = ""
        for l in entry.findall("a:link", ns):
            if l.attrib.get("rel") == "alternate":
                link = l.attrib.get("href", ""); break
        year = (entry.findtext("a:published", default="", namespaces=ns) or "")[:4]
        out.append(_mk_ref(title, link, "arXiv", year, None, "arXiv"))
    return out

def _wikipedia(query: str, n: int) -> List[Dict]:
    url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(query)}&limit={n}&namespace=0&format=json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    titles, urls = data[1], data[3]
    out = []
    for t, u in zip(titles, urls):
        out.append(_mk_ref(t, u, "Wikipedia", None, None, "Wikipedia"))
    return out

# ---------- Closed/paid providers (commented out; add keys to enable) ----------
# Example: Bing Web Search (Azure)
# def _bing_web(query: str, n: int) -> List[Dict]:
#     key = os.getenv("BING_SUBSCRIPTION_KEY")
#     if not key: return []
#     url = f"https://api.bing.microsoft.com/v7.0/search?q={quote_plus(query)}&count={n}"
#     r = requests.get(url, headers={"Ocp-Apim-Subscription-Key": key}, timeout=10); r.raise_for_status()
#     items = r.json().get("webPages", {}).get("value", [])
#     return [_mk_ref(it.get("name"), it.get("url"), "web", None, None, "Bing") for it in items]

# Example: Google Custom Search (CSE)
# def _google_cse(query: str, n: int) -> List[Dict]:
#     key = os.getenv("GOOGLE_CSE_KEY"); cx = os.getenv("GOOGLE_CSE_CX")
#     if not (key and cx): return []
#     url = f"https://www.googleapis.com/customsearch/v1?q={quote_plus(query)}&num={n}&key={key}&cx={cx}"
#     r = requests.get(url, timeout=10); r.raise_for_status()
#     items = r.json().get("items", [])
#     return [_mk_ref(it.get("title"), it.get("link"), it.get("displayLink"), None, None, "GoogleCSE") for it in items]

def basic_search(query: str, limit: int = 6) -> List[Dict]:
    """Aggregate providers; de-duplicate by DOI/URL/title; return <= limit."""
    if not query or not query.strip():
        return []
    # Try more scholarly sources first, then general
    results: List[Dict] = []
    results += _openalex(query, limit)
    results += _crossref(query, limit)
    results += _arxiv(query, limit)
    results += _wikipedia(query, max(2, limit // 2))
    # Optionally include closed providers (uncomment when keys are set)
    # results += _bing_web(query, limit)
    # results += _google_cse(query, limit)
    return _dedupe(results, limit)
