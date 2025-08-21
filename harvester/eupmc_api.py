from __future__ import annotations
import httpx, logging, re

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

def _ensure_pmcid(pmcid: str) -> str:
    pmcid = (pmcid or "").strip()
    if not pmcid:
        return pmcid
    return pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"

def fetch_fulltext_jats(pmcid: str, timeout: float = 60.0) -> str | None:
    """
    Fetch JATS full text for a PMCID.
    Tries Europe PMC first, then falls back to NCBI PMC OAI-PMH.
    Returns the JATS XML as a string (or None if unavailable).
    """
    import httpx, logging
    log = logging.getLogger(__name__)
    pmcid = _ensure_pmcid(pmcid)
    if not pmcid:
        return None

    # 1) Europe PMC: /webservices/rest/{PMCID}/fullTextXML
    url_epmc = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        r = httpx.get(url_epmc, headers={"Accept": "application/xml"}, timeout=timeout, follow_redirects=True)
        if r.status_code == 200 and r.text and "<article" in r.text:
            return r.text
    except Exception as e:
        log.warning("[EUPMC] fullTextXML fetch error for %s: %s", pmcid, e)

    # 2) Fallback: NCBI PMC OAI-PMH → extract <article> … </article>
    url_oai = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&metadataPrefix=pmc&identifier={pmcid}"
    try:
        r = httpx.get(url_oai, headers={"Accept": "application/xml"}, timeout=timeout, follow_redirects=True)
        if r.status_code == 200 and r.text:
            m = re.search(r"(<article[\s\S]*?</article>)", r.text, flags=re.I)
            if m:
                return m.group(1)
            # as a last resort, return the entire OAI response
            return r.text
    except Exception as e:
        log.warning("[PMC OAI] fetch error for %s: %s", pmcid, e)

    return None
