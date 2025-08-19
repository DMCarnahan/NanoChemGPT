import argparse, json, yaml, httpx, re
from pathlib import Path
from tqdm import tqdm
from harvester.utils import ensure_dir, write_json, safe_slug
from harvester.arxiv_api import search_arxiv
from harvester.eupmc_api import search_eupmc, fetch_fulltext_jats
from harvester.unpaywall_api import unpaywall_lookup
from harvester.grobid_client import pdf_to_tei
from harvester.tei_utils import tei_to_sections, filter_methods_sections as filt_tei
from harvester.jats_utils import jats_to_sections, filter_methods_sections as filt_jats
from miner.runtime import get_miner, extract_procedure

miner = get_miner()

ACTION_WORDS = [
    "synthesize", "synthesise", "prepare", "mix", "stir", "disperse", "dissolve", "add",
    "deposit", "coat", "spin coat", "spin-coat", "drop cast", "drop-cast", "anneal",
    "calcine", "dry", "age", "wash", "rinse", "centrifuge", "filter", "grind",
    "heat", "cool", "reflux", "sonicate", "stirred", "heated", "co-precipitate",
    "precipitate", "hydrothermal", "solvothermal", "sol-gel", "autoclave", "spray", "etch"
]

NUM_RX    = r"(?:\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?|\d{1,3}(?:,\d{3})+)"
VOLUME_U  = r"(?:μL|µL|uL|mL|L)"
MASS_U    = r"(?:μg|µg|ug|mg|g|kg)"
MOL_U     = r"(?:μmol|µmol|umol|mmol|mol)"
LENGTH_U  = r"(?:nm|μm|µm|mm|cm)"
UNIT_RX   = rf"(?:{VOLUME_U}|{MASS_U}|{MOL_U}|{LENGTH_U})"

TEMP_RX   = r"(?:\b\d{1,3}\s?°\s?[CF]\b|\b\d+(?:\.\d+)?\s?(?:K|°C|°F)\b)"
TIME_RX   = r"(?:\b\d+(?:\.\d+)?\s?(?:s|sec|secs|seconds|min|mins|h|hr|hrs|hours|day|days)\b)"
SPEED_RX  = r"(?:\b\d+(?:\.\d+)?\s?(?:rpm|r\.?\s?min(?:-1|⁻¹)?)\b)"
CONC_RX   = r"(?:\b\d+(?:\.\d+)?\s?(?:M|mM|μM|µM|uM|%|wt%|vol%|v/v|w/w)\b)"

ACTION_RX = re.compile(r"\b(" + "|".join(map(re.escape, ACTION_WORDS)) + r")(?:ed|ing|s)?\b", re.I)
NUMUNIT_RX= re.compile(rf"{NUM_RX}\s*{UNIT_RX}", re.I)
TEMP_RE   = re.compile(TEMP_RX, re.I)
TIME_RE   = re.compile(TIME_RX, re.I)
SPEED_RE  = re.compile(SPEED_RX, re.I)
CONC_RE   = re.compile(CONC_RX, re.I)

def split_paragraphs(section_text: str):
    """Split text into paragraphs, preferring double newlines, else chunk by sentence."""
    if not section_text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    if paras:
        return paras
    # fallback: chunk by sentences ~300–800 chars
    sents = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", section_text)
    chunk, buf = [], ""
    for s in sents:
        if len(buf) + len(s) < 600:
            buf = (buf + " " + s).strip()
        else:
            if buf: chunk.append(buf)
            buf = s
    if buf: chunk.append(buf)
    return chunk

def score_paragraph(p: str) -> float:
    """Score a paragraph for procedural content."""
    if not p or len(p) < 80:
        return 0.0
    s  = 2.0 * len(ACTION_RX.findall(p))
    s += 1.5 * len(NUMUNIT_RX.findall(p))
    s += 1.0 * len(TEMP_RE.findall(p))
    s += 0.8 * len(TIME_RE.findall(p))
    s += 0.8 * len(SPEED_RE.findall(p))
    s += 0.8 * len(CONC_RE.findall(p))
    if re.search(r"^(To\s+|\bwas\b\s+(?:prepared|synthesized|deposited))", p, re.I):
        s += 0.8
    return s

def fallback_methods_from_sections(sections: list[dict], top_k: int = 5, min_score: float = 2.0):
    """When no explicit Methods/Experimental found, pick top-scoring procedural-looking paragraphs."""
    cands = []
    for sec in sections or []:
        for para in split_paragraphs(sec.get("text", "")):
            sc = score_paragraph(para)
            if sc >= min_score:
                cands.append((sc, {"heading": sec.get("heading", ""), "text": para}))
    cands.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in cands[:top_k]]

def fetch_pdf(url):
    """Fetch PDF content from a URL, following redirects."""
    try:
        r = httpx.get(url, timeout=60.0, follow_redirects=True)
        if r.status_code == 200 and 'application/pdf' in r.headers.get('content-type', ''):
            return r.content
    except Exception as e:
        print(f"[fetch_pdf] Error fetching {url}: {e}")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg['out_dir'])
    ensure_dir(out_dir)

    all_meta = []
    for q in cfg['queries']:
        all_meta += search_arxiv(q, cfg['max_results_per_source'])
        all_meta += search_eupmc(q, cfg['since_year'], cfg['max_results_per_source'])

    seen, dedup = set(), []
    for m in all_meta:
        key = m.get('doi') or m.get('arxiv_id') or m.get('pmcid') or m.get('title')
        if key and key not in seen:
            seen.add(key)
            dedup.append(m)

    bundle = out_dir / 'bundle.jsonl'
    with open(bundle, 'w', encoding='utf-8') as fout:
        for rec in tqdm(dedup, desc='Processing'):
            paper = {
                'paper_id': rec.get('doi') or rec.get('arxiv_id') or rec.get('pmcid'),
                'title': rec.get('title', ''),
                'authors': [{'name': a} for a in rec.get('authors', [])],
                'doi': rec.get('doi'),
                'source': rec.get('source'),
                'urls': {'pdf': rec.get('pdf_url')},
                'license': rec.get('license'),
                'access_route': rec.get('access_route', 'unknown'),
                'sections': [],
                'extractions': {'methods_paragraphs': []}
            }

            methods = []
            all_sections = []

            # --- JATS (PMC) path
            if rec.get('pmcid'):
                jats = fetch_fulltext_jats(rec['pmcid'])
                if jats:
                    secs = jats_to_sections(jats)
                    all_sections = secs
                    methods = filt_jats(secs)

            # --- Unpaywall -> PDF -> GROBID if still nothing
            if not methods:
                if rec.get('doi') and cfg.get('unpaywall_email'):
                    up = unpaywall_lookup(rec['doi'], cfg['unpaywall_email'])
                    if up and up.get('oa_locations'):
                        paper['license'] = {'type': (up.get('best_oa_location') or {}).get('license') or 'oa'}
                        loc = up.get('best_oa_location') or up['oa_locations'][0]
                        rec['pdf_url'] = rec.get('pdf_url') or loc.get('url_for_pdf') or loc.get('url')
                if rec.get('pdf_url'):
                    pdf = fetch_pdf(rec['pdf_url'])
                    if pdf:
                        try:
                            tei = pdf_to_tei(cfg['grobid_url'], pdf)
                            secs = tei_to_sections(tei)
                            if not all_sections:
                                all_sections = secs
                            methods = filt_tei(secs)
                        except Exception as e:
                            print(f"[GROBID] Error: {e}")

            # --- Fallback: pick top-scoring procedural paragraphs if still empty
            if not methods and all_sections:
                methods = fallback_methods_from_sections(all_sections, top_k=5, min_score=2.0)

            # Save chosen sections (traceability)
            paper['sections'] = methods[:5]

            # Run the miner on those paragraphs
            paras = [s['text'] for s in methods]
            if paras:
                mp = []
                for ptxt in paras:
                    ann = extract_procedure(ptxt)
                    mp.append({
                        "text": ptxt,
                        "operations": ann.get("operations", []),
                        "expanded": ann.get("expanded", [])
                    })
                paper['extractions']['methods_paragraphs'] = mp
            else:
                paper['extractions']['methods_paragraphs'] = []

            pid = safe_slug(paper['paper_id'] or paper['title'][:40])
            write_json(out_dir / f'{pid}.json', paper)
            fout.write(json.dumps(paper, ensure_ascii=False) + '\n')

    print(f'Done. Bundle: {bundle}')

if __name__ == '__main__':
    main()
