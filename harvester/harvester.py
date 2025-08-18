import argparse, json, yaml, httpx
from pathlib import Path
from tqdm import tqdm
from utils import ensure_dir, write_json, safe_slug
from arxiv_api import search_arxiv
from eupmc_api import search_eupmc, fetch_fulltext_jats
from unpaywall_api import unpaywall_lookup
from grobid_client import pdf_to_tei
from tei_utils import tei_to_sections, filter_methods_sections as filt_tei
from jats_utils import jats_to_sections, filter_methods_sections as filt_jats
from miner import load_pipeline, run_ner_link

SPACY_MODEL_DIR = Path("./miner/ner_model/model-best")
HEURISTIC_LINKER_PATH = Path("./miner/heuristic_linker.py")

def fetch_pdf(url):
    try:
        r=httpx.get(url, timeout=60.0, follow_redirects=True)
        if r.status_code==200 and 'application/pdf' in r.headers.get('content-type',''):
            return r.content
    except Exception:
        return None
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    out_dir=Path(cfg['out_dir']); ensure_dir(out_dir)
    nlp, link_doc = load_pipeline(SPACY_MODEL_DIR, HEURISTIC_LINKER_PATH)
    all_meta=[]
    for q in cfg['queries']:
        all_meta+=search_arxiv(q, cfg['max_results_per_source'])
        all_meta+=search_eupmc(q, cfg['since_year'], cfg['max_results_per_source'])
    seen=set(); dedup=[]
    for m in all_meta:
        key = m.get('doi') or m.get('arxiv_id') or m.get('pmcid') or m.get('title')
        if key and key not in seen: seen.add(key); dedup.append(m)
    bundle=out_dir/'bundle.jsonl'
    with open(bundle,'w',encoding='utf-8') as fout:
        for rec in tqdm(dedup, desc='Processing'):
            paper={'paper_id': rec.get('doi') or rec.get('arxiv_id') or rec.get('pmcid'), 'title': rec.get('title',''), 'authors':[{'name':a} for a in rec.get('authors',[])], 'doi':rec.get('doi'), 'source':rec.get('source'), 'urls':{'pdf':rec.get('pdf_url')}, 'license':rec.get('license'), 'access_route':rec.get('access_route','unknown'), 'sections':[], 'extractions':{'methods_paragraphs':[]}}
            methods=[]
            if rec.get('pmcid'):
                jats=fetch_fulltext_jats(rec['pmcid'])
                if jats:
                    secs=jats_to_sections(jats); methods=filt_jats(secs)
            if not methods:
                if rec.get('doi') and cfg.get('unpaywall_email'):
                    up=unpaywall_lookup(rec['doi'], cfg['unpaywall_email'])
                    if up and up.get('oa_locations'):
                        paper['license']={'type': (up.get('best_oa_location') or {}).get('license') or 'oa'}
                        loc=up.get('best_oa_location') or up['oa_locations'][0]
                        rec['pdf_url']=rec.get('pdf_url') or loc.get('url_for_pdf') or loc.get('url')
                if rec.get('pdf_url'):
                    pdf=fetch_pdf(rec['pdf_url'])
                    if pdf:
                        try:
                            tei=pdf_to_tei(cfg['grobid_url'], pdf)
                            secs=tei_to_sections(tei); methods=filt_tei(secs)
                        except Exception:
                            pass
            paper['sections']=methods[:5]
            paras=[s['text'] for s in methods]
            if paras:
                paper['extractions']['methods_paragraphs']=run_ner_link(nlp, link_doc, paras)
            pid=safe_slug(paper['paper_id'] or paper['title'][:40])
            write_json(out_dir/f'{pid}.json', paper)
            fout.write(json.dumps(paper, ensure_ascii=False)+'\n')
    print(f'Done. Bundle: {bundle}')

if __name__=='__main__':
    main()
