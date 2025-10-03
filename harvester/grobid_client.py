import httpx


def pdf_to_tei(url, pdf_bytes):
    u = url.rstrip("/") + "/api/processFulltextDocument"
    files = {"input": ("paper.pdf", pdf_bytes, "application/pdf")}
    data = {"consolidateHeader": "1", "consolidateCitations": "0"}
    r = httpx.post(u, files=files, data=data, timeout=120.0)
    r.raise_for_status()
    return r.text
