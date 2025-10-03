# harvester/oa_resolver.py
import os
import urllib.parse

import requests

UNPAYWALL = "https://api.unpaywall.org/v2/"
OPENALEX = "https://api.openalex.org/works/"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

UA = "NanoChemGPT-Harvester/1.0 (+https://nanochemgpt-production.up.railway.app/)"
TIMEOUT = float(os.getenv("OA_TIMEOUT", "12"))


def _get(session, url, **kw):
    kw.setdefault("timeout", TIMEOUT)
    kw.setdefault("headers", {"User-Agent": UA})
    r = session.get(url, **kw)
    r.raise_for_status()
    return r


def unpaywall_get(doi: str, email: str, session=None) -> dict:
    if not doi:
        return {}
    session = session or requests.Session()
    url = UNPAYWALL + urllib.parse.quote(doi)
    r = _get(session, url, params={"email": email})
    return r.json()


def openalex_get_by_doi(doi: str, session=None) -> dict:
    if not doi:
        return {}
    session = session or requests.Session()
    url = OPENALEX + f"doi:{urllib.parse.quote(doi.lower())}"
    r = _get(session, url, params={"mailto": os.getenv("OPENALEX_MAILTO", "")})
    return r.json()


def eupmc_find_by_doi(doi: str, session=None) -> dict:
    session = session or requests.Session()
    q = f"EXT_ID:{doi}"
    r = _get(session, EPMC, params={"query": q, "format": "json", "resultType": "core"})
    return r.json()


def eupmc_best_pdf(json_obj: dict) -> str:
    """
    Returns a direct PDF URL if available for open-access Europe PMC record.
    """
    try:
        results = json_obj.get("resultList", {}).get("result", [])
        for rec in results:
            if str(rec.get("isOpenAccess", "false")).lower() != "true":
                continue
            # Prefer PDF link in fullTextUrlList
            for ft in rec.get("fullTextUrlList", {}).get("fullTextUrl", []):
                if (
                    ft.get("availability") == "Open access"
                    and ft.get("documentStyle") == "pdf"
                ):
                    return ft.get("url") or ""
    except Exception:
        pass
    return ""


def resolve_oa(doi: str, session=None) -> dict:
    """
    Returns:
      {
        "is_oa": bool,
        "source": "unpaywall"|"eupmc"|"openalex",
        "host_type": "publisher"|"repository"|None,
        "license": str|None,
        "version": str|None,   # publishedVersion/acceptedVersion/submittedVersion
        "url": str|None,       # preferred URL (html if pdf missing)
        "pdf_url": str|None    # direct PDF if known
      }
    """
    session = session or requests.Session()
    email = os.getenv("UNPAYWALL_EMAIL", "")
    out = {
        "is_oa": False,
        "source": None,
        "host_type": None,
        "license": None,
        "version": None,
        "url": None,
        "pdf_url": None,
    }

    # 1) Unpaywall
    if email:
        try:
            u = unpaywall_get(doi, email, session=session)
            if u.get("is_oa"):
                out.update(
                    is_oa=True,
                    source="unpaywall",
                    host_type=(u.get("best_oa_location") or {}).get("host_type"),
                    license=(u.get("best_oa_location") or {}).get("license"),
                    version=(u.get("best_oa_location") or {}).get("version"),
                    url=(u.get("best_oa_location") or {}).get("url"),
                    pdf_url=(u.get("best_oa_location") or {}).get("url_for_pdf"),
                )
                if out["pdf_url"] and out["pdf_url"].startswith("http"):
                    return out
        except Exception:
            pass

    # 2) Europe PMC by DOI
    try:
        ej = eupmc_find_by_doi(doi, session=session)
        pdf = eupmc_best_pdf(ej)
        if pdf:
            out.update(
                is_oa=True,
                source="eupmc",
                host_type="repository",
                version="publishedVersion",
                pdf_url=pdf,
                url=pdf,
            )
            return out
    except Exception:
        pass

    # 3) OpenAlex fallback
    try:
        w = openalex_get_by_doi(doi, session=session)
        oa = w.get("open_access") or {}
        if oa.get("is_oa"):
            out.update(
                is_oa=True,
                source="openalex",
                host_type=oa.get("oa_status"),  # not strictly host_type, but useful
                url=oa.get("oa_url"),
            )
    except Exception:
        pass

    return out
