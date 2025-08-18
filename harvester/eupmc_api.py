import httpx
EPMC_SEARCH='https://www.ebi.ac.uk/europepmc/webservices/rest/search'
EPMC_FULL='https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'

def search_eupmc(q,since_year=2018,max_results=200):
    q=f'({q}) AND (FIRST_PDATE:[{since_year}-01-01 TO 3000-12-31]) AND (OPEN_ACCESS:y)'
    r=httpx.get(EPMC_SEARCH, params={'query':q,'format':'json','pageSize':1000}, timeout=45.0); r.raise_for_status(); js=r.json()
    hits=js.get('resultList',{}).get('result',[]) or []
    out=[]
    for it in hits[:max_results]:
        out.append({'source':'eupmc','pmcid':it.get('pmcid'),'title':it.get('title',''),'authors':[a.get('fullName','') for a in it.get('authorList',{}).get('author',[])], 'doi':it.get('doi'), 'pdf_url':it.get('pdfUrl'), 'license':{'type':it.get('license','unknown')}, 'access_route':'oa'})
    return out

def fetch_fulltext_jats(pmcid):
    if not pmcid: return None
    r=httpx.get(EPMC_FULL.format(pmcid=pmcid), timeout=45.0)
    return r.text if r.status_code==200 else None
