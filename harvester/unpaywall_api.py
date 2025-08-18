import httpx
UNPAYWALL='https://api.unpaywall.org/v2/'

def unpaywall_lookup(doi,email):
    if not doi: return None
    r=httpx.get(UNPAYWALL+doi, params={'email':email}, timeout=30.0)
    return r.json() if r.status_code==200 else None
