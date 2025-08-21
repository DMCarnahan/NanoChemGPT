from __future__ import annotations
import httpx, logging

log = logging.getLogger(__name__)

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

def build_eupmc_query(q: str, since_year: int) -> str:
    return f"({q}) AND (FIRST_PDATE:[{since_year}-01-01 TO 3000-12-31]) AND (OPEN_ACCESS:y)"

def search_eupmc(q: str, since_year: int, max_results: int = 200) -> list[dict]:
    query = build_eupmc_query(q, since_year)
    params = {
        "query": query,
        "format": "json",        # JSON output
        "resultType": "lite",    # sufficient for metadata harvest
        "pageSize": min(max_results, 1000),
        "synonym": "false",      
    }
    try:
        r = httpx.get(
            EPMC_SEARCH,
            params=params,
            headers={"Accept": "application/json"},
            timeout=45.0,
            follow_redirects=True,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.warning("[EUPMC] HTTP %s at %s", e.response.status_code, e.request.url)
        return []
    except Exception as e:
        log.warning("[EUPMC] request error: %s", e)
        return []

    js = r.json()
    results = (js.get("resultList") or {}).get("result") or []

    out = []
    for it in results:
        pmcid = it.get("pmcid")
        doi   = it.get("doi")
        title = it.get("title") or ""
        authors = (it.get("authorString") or "").split(", ") if it.get("authorString") else []
        is_oa = (it.get("isOpenAccess") == "Y")
        pdf_url = it.get("pdfUrl")
        if not pdf_url and pmcid:
            # Europe/NCBI PMC PDFs follow this pattern
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"

        out.append({
            "title": title,
            "doi": doi,
            "pmcid": pmcid,
            "authors": authors,
            "source": "eupmc",
            "pdf_url": pdf_url,
            "license": {"type": "oa" if is_oa else "unknown"},
            "access_route": "oa" if is_oa else "unknown",
        })
    return out
