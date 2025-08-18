import httpx, xml.etree.ElementTree as ET
from urllib.parse import urlencode
ARXIV_API='https://export.arxiv.org/api/query'

def search_arxiv(query,max_results=100):
    url = ARXIV_API+'?'+urlencode({'search_query':query,'start':0,'max_results':max_results})
    r=httpx.get(url, timeout=30.0); r.raise_for_status()
    root=ET.fromstring(r.text); ns={'a':'http://www.w3.org/2005/Atom'}
    out=[]
    for e in root.findall('a:entry',ns):
        pid=e.findtext('a:id',namespaces=ns)
        title=e.findtext('a:title',namespaces=ns) or ''
        summary=e.findtext('a:summary',namespaces=ns) or ''
        authors=[a.findtext('a:name',namespaces=ns) for a in e.findall('a:author',ns)]
        pdf=None
        for link in e.findall('a:link',ns):
            if link.attrib.get('title')=='pdf' or link.attrib.get('type')=='application/pdf':
                pdf=link.attrib.get('href')
        out.append({'source':'arxiv','arxiv_id':(pid or '').split('/')[-1],'title':title.strip(),'abstract':summary.strip(),'authors':authors,'pdf_url':pdf,'access_route':'oa','license':{'type':'arXiv-OA'}})
    return out
