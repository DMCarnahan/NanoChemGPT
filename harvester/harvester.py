import argparse, json, yaml, httpx, re, os, sys, html_text
from pathlib import Path
from tqdm import tqdm
from utils import ensure_dir, write_json, safe_slug
from arxiv_api import search_arxiv
from eupmc_api import search_eupmc, fetch_fulltext_jats
from unpaywall_api import unpaywall_lookup
from grobid_client import pdf_to_tei
from tei_utils import tei_to_sections, filter_methods_sections as filt_tei
from jats_utils import jats_to_sections, filter_methods_sections as filt_jats
from miner.runtime import get_miner
from oa_resolver import resolve_oa
from enhanced_relevance import enhance_harvester_relevance

miner = get_miner(nlp_model= "SPACY_MODEL")

import logging
logger = logging.getLogger(__name__)

UA = "NanoChemGPT-Harvester/1.0 (+https://nanochemgpt-production.up.railway.app/)"
TIMEOUT = float(os.getenv("OA_TIMEOUT", "12"))
# ---------------------- Normalization helpers ----------------------
_DOI_RX = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.I)

def _norm_str(x):
    if x is None:
        return ""
    return str(x).strip()

def _norm_doi(x) -> str:
    s = _norm_str(x)
    if not s:
        return ""
    m = _DOI_RX.search(s)
    return m.group(1).lower() if m else ""

def _pmcid_from_any(x: str) -> str:
    s = _norm_str(x)
    if not s:
        return ""
    m = re.search(r'\bPMC\d+\b', s, flags=re.I)
    return m.group(0).upper() if m else ""

def _arxiv_id_from_any(url_or_id: str) -> str:
    s = _norm_str(url_or_id)
    if not s:
        return ""
    m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:\.pdf)?', s, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r'\b([0-9]{4}\.[0-9]{4,5})\b', s)
    return m.group(1) if m else ""

def _pdf_url_guess_arxiv(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""

def _authors_to_list(authors_field):
    if not authors_field:
        return []
    if isinstance(authors_field, list):
        return [str(a).strip() for a in authors_field if str(a).strip()]
    s = str(authors_field)
    parts = re.split(r'\s*;\s*|\s*\band\b\s*|\s*,\s*(?=[A-Z][a-z]+(?:\s|$))', s)
    out = [p.strip() for p in parts if p and len(p.strip()) > 1]
    seen, uniq = set(), []
    for a in out:
        if a not in seen:
            seen.add(a); uniq.append(a)
    return uniq

def _year_from(*vals):
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()           
        if len(s) >= 4 and s[:4].isdigit():
            return s[:4]
        m = re.search(r'(\d{4})', s)
        if m:
            return m.group(1)
    return ""

def _canon_rec(rec: dict) -> dict:
    doi = _norm_doi(rec.get("doi") or rec.get("url") or rec.get("id") or "")
    pmcid = rec.get("pmcid") or ""
    if not pmcid:
        pmcid = _pmcid_from_any((rec.get("id","") + " " + rec.get("url","")).strip())
    arxiv_id = rec.get("arxiv_id") or ""
    if not arxiv_id:
        arxiv_id = _arxiv_id_from_any((rec.get("id","") + " " + rec.get("url","")).strip())
    title = rec.get("title") or ""
    url = rec.get("pdf_url") or rec.get("url") or ""
    pdf_url = rec.get("pdf_url") or ""
    if not pdf_url and arxiv_id:
        pdf_url = _pdf_url_guess_arxiv(arxiv_id)
    if not pdf_url and url and url.lower().endswith(".pdf"):
        pdf_url = url
    authors_list = _authors_to_list(rec.get("authors"))
    year = _year_from(rec.get("year"), rec.get("pubYear"), rec.get("published"))
    source = rec.get("source") or ""
    return {
        "doi": doi,
        "pmcid": pmcid,
        "arxiv_id": arxiv_id,
        "title": title,
        "authors_list": authors_list,
        "year": year,
        "url": rec.get("url") or "",
        "pdf_url": pdf_url,
        "license": rec.get("license"),
        "access_route": rec.get("access_route", "unknown"),
        "source": source or ("arxiv" if arxiv_id else ("eupmc" if pmcid else rec.get("source","")))
    }

def _dedup_key(c):
    return c["doi"] or c["pmcid"] or c["arxiv_id"] or (c["title"][:200].lower())
# ------------------------------------------------------------------



def harvest_one_record(rec):
    """
    rec should contain at least: title, authors, year, doi
    """
    doi = (rec.get("doi") or "").strip().lower()
    meta = {"doi": doi, "title": rec.get("title"), "year": rec.get("year"), "authors": rec.get("authors")}

    # Resolve OA copy
    oa = resolve_oa(doi)
    meta.update({"oa": oa})
    if not oa.get("is_oa"):
        return {"meta": meta, "text": None, "why": "no_oa"}

    # Fetch
    url = oa.get("pdf_url") or oa.get("url")
    if not url:
        return {"meta": meta, "text": None, "why": "oa_url_missing"}

    text = fetch_and_extract_text(url, license_hint=oa.get("license"))
    return {"meta": meta, "text": text, "why": "ok"}

def fetch_and_extract_text(url: str, license_hint: str | None = None) -> str:
    import requests
    from pdfminer.high_level import extract_text as pdf_extract_text
    headers = {"User-Agent": UA, "Referer": "https://example.org"}
    with requests.get(url, headers=headers, timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            # Write to tmp + parse PDF
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                for chunk in r.iter_content(1<<16):
                    f.write(chunk)
                fname = f.name
            try:
                txt = pdf_extract_text(fname) or ""
            finally:
                try: os.remove(fname)
                except Exception: pass
            return txt
        else:
            # HTML article page (publisher OA or repo): extract readable text
            html = r.text
            return html_text.extract_text(html)  


def extract_plain_text_from_pdf(pdf_bytes: bytes) -> str:
    # pdfminer-only fallback, no fitz and no is_pdf_bytes
    if not isinstance(pdf_bytes, (bytes, bytearray)) or len(pdf_bytes) < 5:
        return ""
    from pdfminer.high_level import extract_text as pdf_extract_text
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        fname = f.name
    try:
        return pdf_extract_text(fname) or ""
    finally:
        try: os.remove(fname)
        except Exception: pass



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

_SECTION_RX = re.compile(
    r"(?im)^\s*(materials and methods|methods?|experimental)\b.*?"
    r"(?=^\s*(results?|discussion|conclusion|acknowledg(e)?ments?)\b|\Z)", re.S | re.M
)
def fallback_methods_from_text(plain_text: str) -> str | None:
    m = _SECTION_RX.search(plain_text or "")
    return m.group(0).strip() if m else None

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

def main():
    import logging
    from urllib.parse import urljoin

    logger = logging.getLogger(__name__)
    USE_GROBID = os.getenv("USE_GROBID", "1") not in {"0", "false", "False"}

    # -------- helpers local to main() --------
    PDF_MAGIC = b"%PDF-"
    PDF_HREF_RX = re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.I)

    def is_pdf_bytes(b: bytes) -> bool:
        return isinstance(b, (bytes, bytearray)) and len(b) > 4 and b[:5] == PDF_MAGIC

    def fetch_pdf_or_html(url: str, timeout: float = 60.0):
        """Return (kind, payload, final_url). kind ∈ {'pdf','html','other'}."""
        try:
            r = httpx.get(url, timeout=timeout, follow_redirects=True)
        except Exception as e:
            logger.warning("[fetch] error %s", e)
            return "other", None, url
        ctype = (r.headers.get("content-type") or "").lower()
        if "application/pdf" in ctype and is_pdf_bytes(r.content):
            return "pdf", r.content, str(r.url)
        if "text/html" in ctype:
            return "html", r.text, str(r.url)
        if is_pdf_bytes(r.content):  # servers sometimes mislabel
            return "pdf", r.content, str(r.url)
        return "other", None, str(r.url)

    def discover_pdf_in_html(html: str, base_url: str) -> str | None:
        m = PDF_HREF_RX.search(html or "")
        if not m:
            return None
        return urljoin(base_url, m.group(1))
    # -----------------------------------------

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    out_dir = Path(cfg['out_dir'])
    ensure_dir(out_dir)

    # Coerce types for numeric config values
    cfg['max_results_per_source'] = int(cfg.get('max_results_per_source', 50))
    cfg['since_year'] = int(cfg.get('since_year', 2000))

    all_meta = []
    for q in cfg['queries']:
        for r in search_arxiv(q, cfg['max_results_per_source']):
            r['source'] = 'arxiv'; all_meta.append(r)
        for r in search_eupmc(q, cfg['since_year'], cfg['max_results_per_source']):
            r['source'] = 'eupmc'; all_meta.append(r)

    # Normalize all records to a common schema
    canon = [_canon_rec(r) for r in all_meta]

    # De-duplicate by DOI → PMCID → arXiv ID → title
    seen, dedup = set(), []
    for c in canon:
        key = _dedup_key(c)
        if key and key not in seen:
            seen.add(key)
            dedup.append(c)

    # Apply enhanced relevance filtering if enabled
    enable_enhanced_relevance = cfg.get('enable_enhanced_relevance', False)
    if enable_enhanced_relevance:
        logger.info(f"Applying enhanced relevance filtering to {len(dedup)} papers...")
        try:
            # Convert to format expected by enhanced_relevance
            papers_for_filtering = []
            for rec in dedup:
                paper_dict = {
                    "title": rec.get("title", ""),
                    "abstract": rec.get("abstract", ""),  # May be empty for many papers
                    "journal": rec.get("journal", ""),
                    "year": rec.get("year"),
                    "doi": rec.get("doi"),
                    "isOpenAccess": rec.get("access_route") not in ["unknown", "closed"],
                    "text": "",  # Will be populated later during processing
                    "keywords": rec.get("keywords", [])
                }
                papers_for_filtering.append(paper_dict)
            
            # Apply enhanced relevance with current config
            relevance_config = {
                "min_year": cfg.get("min_year", cfg.get("since_year", 2018)),
                "quality_threshold": cfg.get("quality_threshold", 0.4),
                "max_papers": cfg.get("max_papers", len(dedup)),
                "queries": cfg.get("queries", [])
            }
            
            enhanced_papers = enhance_harvester_relevance(papers_for_filtering, relevance_config)
            
            # Map back to original dedup format, preserving enhanced metadata
            relevance_map = {i: paper for i, paper in enumerate(enhanced_papers)}
            filtered_dedup = []
            
            for i, rec in enumerate(dedup):
                if i < len(enhanced_papers):
                    # Add relevance metadata to the record
                    rec["relevance_score"] = enhanced_papers[i].get("relevance_score", 0.0)
                    rec["relevance_breakdown"] = enhanced_papers[i].get("relevance_breakdown", {})
                    rec["relevance_reasons"] = enhanced_papers[i].get("relevance_reasons", [])
                    filtered_dedup.append(rec)
            
            dedup = filtered_dedup
            logger.info(f"Enhanced relevance filtering retained {len(dedup)} papers")
            
        except Exception as e:
            logger.warning(f"Enhanced relevance filtering failed: {e}. Continuing with original papers.")
    
    else:
        logger.info(f"Enhanced relevance filtering disabled. Processing all {len(dedup)} papers.")

    # Prepare bundle file
    bundle = Path(cfg.get("out_bundle", out_dir / "bundle.jsonl"))
    written = 0
    with open(bundle, "w", encoding="utf-8") as fout:
        for rec in tqdm(dedup, desc='Processing'):
            paper = {
                'paper_id': rec.get('doi') or rec.get('arxiv_id') or rec.get('pmcid') or rec.get('title','')[:40],
                'title': rec.get('title', ''),
                'authors': [{'name': a} for a in (rec.get('authors_list') or [])],
                'doi': rec.get('doi'),
                'source': rec.get('source'),
                'urls': {'pdf': rec.get('pdf_url') or ''},
                'license': rec.get('license'),
                'access_route': rec.get('access_route', 'unknown'),
                'sections': [],
                'extractions': {'methods_paragraphs': []},
                'relevance': {
                    'score': rec.get('relevance_score', 0.0),
                    'breakdown': rec.get('relevance_breakdown', {}),
                    'reasons': rec.get('relevance_reasons', [])
                },
                'meta': {
                    'title': rec.get('title', ''),
                    'doi': rec.get('doi') or '',
                    'url': rec.get('url') or '',
                    'pdf_url': rec.get('pdf_url') or '',
                    'year': rec.get('year') or '',
                    'authors': rec.get('authors_list') or []
                }
            }

            methods: list[dict] = []
            all_sections: list[dict] = []
            plain: str = ""

            # --- JATS (PMC) path
            if rec.get('pmcid'):
                jats = fetch_fulltext_jats(rec['pmcid'])
                if jats:
                    secs = jats_to_sections(jats)
                    all_sections = secs
                    methods = filt_jats(secs)

            # --- Unpaywall -> PDF -> (GROBID or plain-text fallback)
            if not methods:
                # Try to improve PDF URL via Unpaywall
                if rec.get('doi'):
                    try:
                        oa = resolve_oa(rec['doi'])
                        rec['access_route'] = (oa.get('host_type') or 'unknown')
                        rec['license'] = oa.get('license') or rec.get('license')
                        if oa.get('is_oa'):
                            rec['pdf_url'] = rec.get('pdf_url') or oa.get('pdf_url') or oa.get('url')
                    except Exception:
                        pass

                if rec.get('pdf_url') or rec.get('url'):
                    start_url = rec.get('pdf_url') or rec.get('url')
                    pdf_bytes = None
                    kind, payload, final_url = fetch_pdf_or_html(start_url)
                    if kind == 'pdf':
                        pdf_bytes = payload
                    elif kind == 'html':
                        alt_pdf = discover_pdf_in_html(payload, final_url)
                        if alt_pdf:
                            k2, p2, _ = fetch_pdf_or_html(alt_pdf)
                            if k2 == 'pdf':
                                pdf_bytes = p2

                    if pdf_bytes and is_pdf_bytes(pdf_bytes):
                        # Preferred: GROBID (if enabled)
                        if USE_GROBID:
                            try:
                                tei = pdf_to_tei(cfg['grobid_url'], pdf_bytes)
                                secs = tei_to_sections(tei)
                                if not all_sections:
                                    all_sections = secs
                                methods = filt_tei(secs)
                            except Exception as e:
                                logger.warning("[GROBID] Error; will try plain-text fallback: %s", e)

                        # Fallback: extract plain text + regex pick
                        if not methods:
                            try:
                                plain = extract_plain_text_from_pdf(pdf_bytes)  # may not exist → NameError
                            except NameError:
                                plain = ""
                            except Exception as e:
                                logger.warning("plain-text extraction failed: %s", e)
                                plain = ""

                            if plain:
                                if not all_sections:
                                    all_sections = [{"heading": "fulltext", "text": plain}]
                                found = fallback_methods_from_text(plain)
                                if found:
                                    methods = [{"heading": "(fallback)", "text": found}]

            # --- LAST RESORT: score paragraphs if still empty
            if not methods and (all_sections or plain):
                if not all_sections and plain:
                    all_sections = [{"heading": "fulltext", "text": plain}]
                methods = fallback_methods_from_sections(all_sections, top_k=8, min_score=1.2)

            if not methods:
                meta_bits = []
                if rec.get("abstract"):  # arXiv/EUPMC often have this
                    meta_bits.append(rec["abstract"])
                if rec.get("title"):
                    meta_bits.append(rec["title"])
                meta_text = "\n\n".join([t for t in meta_bits if t])
                if meta_text:
                    all_sections = all_sections or [{"heading": "metadata", "text": meta_text}]
                    methods = fallback_methods_from_sections(all_sections, top_k=5, min_score=1.0)

            # Save chosen sections (traceability)
            paper['sections'] = methods[:5]

            if plain:
                paper['raw'] = plain[:2_000_000]

            # Run the miner on those paragraphs
            paras = [s['text'] for s in methods]
            entities_for_relevance = []
            if paras:
                mp = []
                for ptxt in paras:
                    ann = miner.extract_procedure(ptxt)
                    mp.append({
                        "text": ptxt,
                        "operations": ann.get("operations", []),
                        "expanded": ann.get("expanded", [])
                    })
                    # Collect entities for relevance scoring
                    for op in ann.get("operations", []):
                        for material in op.get("materials", []):
                            entities_for_relevance.append({"label": "MATERIAL", "text": material})
                        for param_key, param_val in op.get("params", {}).items():
                            if param_key in ["temperature", "temp"]:
                                entities_for_relevance.append({"label": "TEMP", "text": str(param_val)})
                            elif param_key in ["time", "duration"]:
                                entities_for_relevance.append({"label": "TIME", "text": str(param_val)})
                            elif param_key in ["amount", "quantity"]:
                                entities_for_relevance.append({"label": "AMOUNT", "text": str(param_val)})
                
                paper['extractions']['methods_paragraphs'] = mp
                paper['extractions']['entities'] = entities_for_relevance
            else:
                paper['extractions']['methods_paragraphs'] = []
                paper['extractions']['entities'] = []

            if not paper.get("raw"):
                raw_source = ""
                if all_sections:
                    raw_source = "\n\n".join(sec.get("text", "") for sec in all_sections if sec.get("text"))
                if not raw_source:
                    meta_bits = []
                    if rec.get("abstract"): meta_bits.append(rec["abstract"])
                    if rec.get("title"):    meta_bits.append(rec["title"])
                    raw_source = "\n\n".join(meta_bits)
                paper["raw"] = (raw_source or "")[:2_000_000]

            pid = safe_slug(paper['paper_id'] or paper['title'][:40])
            write_json(out_dir / f'{pid}.json', paper)
            fout.write(json.dumps(paper, ensure_ascii=False) + '\n')
            written += 1

    print(f'Done. Bundle: {bundle} (wrote {written} records)')  

if __name__ == '__main__':
    main()