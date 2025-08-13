
# NanoChemGPT Flask Application
# --------------------------------
# Main server and API for NanoChemGPT, a chemistry protocol and reasoning assistant.
# Author: [Your Name or Organization]
# License: MIT
# --------------------------------

from __future__ import annotations


import os
import io
import json
import re
import glob
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from functools import lru_cache

from typing import List, Dict, Tuple, Set
import httpx
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, request, jsonify, abort, render_template,
    send_file, make_response
)
from jinja2 import TemplateNotFound
from openai import OpenAI
from werkzeug.utils import secure_filename

# ---- Local Modules ---- #
import vector_store as vs
from converter import validate_step, convert_text_to_robot_ops
from mongo_client import get_db, ping as mongo_ping
from internet_search import search_papers
from DuckDB.duck_searcher import get_duck_searcher


# ---- Paths & Config ---- #
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
BUILTIN_DIR = Path(os.getenv("BUILTIN_DIR", BASE_DIR / "builtin")).resolve()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "/mnt/data/uploads")).resolve()
LOOKUP_UPLOAD_DIR = Path(os.getenv("LOOKUP_UPLOAD_DIR", "/mnt/data/datasets")).resolve()
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "/mnt/data/index")).resolve()


# Ensure all required directories exist
for d in (BUILTIN_DIR, UPLOADS_DIR, LOOKUP_UPLOAD_DIR, VECTORSTORE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---- Flask App Setup ---- #
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.config["JSON_AS_ASCII"] = False  # allow UTF-8


# CSRF setup (optional)
try:
    from flask_wtf.csrf import CSRFProtect, generate_csrf
    app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "change-me")
    app.config['WTF_CSRF_TIME_LIMIT'] = None
    csrf = CSRFProtect(app)
except Exception:
    csrf = None
    def generate_csrf() -> str:
        return ""


@app.context_processor
def inject_csrf_token():
    """Inject CSRF token for Jinja templates."""
    return dict(csrf_token=generate_csrf)


@app.before_request
def _log_req():
    """Log every incoming request."""
    print(f"[req] {request.method} {request.path}")

    
# ---- DuckDB Setup ---- #
def maybe_build_duckdb():
    """Create a .duckdb from Parquet once if DUCKDB_BOOTSTRAP=1 and files exist."""
    if os.getenv("DUCKDB_BOOTSTRAP", "0").lower() not in ("1", "true", "yes"):
        app.logger.info("[duckdb-init] skipped (DUCKDB_BOOTSTRAP not set)")
        return
    parq_glob = os.getenv("LOOKUP_PARQUET_GLOB")
    db_path = os.getenv("LOOKUP_DUCKDB_PATH")
    tbl = os.getenv("LOOKUP_DUCKDB_TABLE", "reactions")
    if not (parq_glob and db_path):
        app.logger.info("[duckdb-init] missing LOOKUP_PARQUET_GLOB or LOOKUP_DUCKDB_PATH")
        return
    matches = glob.glob(parq_glob, recursive=True)
    if not matches:
        app.logger.warning("[duckdb-init] no parquet matched %r", parq_glob)
        return
    if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
        app.logger.info("[duckdb-init] db exists at %s (not rebuilding)", db_path)
        return
    try:
        import duckdb, pathlib
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(db_path)
        con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM read_parquet('{parq_glob}', hive_partitioning=1)")
        con.execute("CHECKPOINT")
        row = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
        rows = row[0] if row else 0
        con.close()
        app.logger.info("[duckdb-init] created %s rows=%d table=%s", db_path, rows, tbl)
    except Exception as e:
        app.logger.warning("[duckdb-init] build failed: %s", e)

maybe_build_duckdb()

LOOKUP = None
try:
    LOOKUP = get_duck_searcher()
    if LOOKUP:
        view = getattr(LOOKUP, "view", None)
        con = getattr(LOOKUP, "con", None)
        if view and con:
            try:
                nrows = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
                app.logger.info("[dataset_search] view '%s' rows=%s", view, nrows)
            except Exception:
                app.logger.info("[dataset_search] DuckDB connected (row count probe failed)")
    else:
        app.logger.warning("[dataset_search] no LOOKUP source configured")
except Exception as e:
    app.logger.warning("[dataset_search] init failed: %s", e)


# ---- OpenAI Client ---- #
load_dotenv()
_no_proxy = httpx.Client(trust_env=False, timeout=120.0)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=_no_proxy)


# ---- Utility Functions ---- #
def _safe_text(x: Any) -> str:
    """Convert any value to string, safely handling None."""
    try:
        return str(x) if x is not None else ""
    except Exception:
        return ""


def _extract_used_markers(*texts: str) -> dict:
    """Find [n] citations and [CTX]/[PARSED]/[DB]/[GEN] tags in text."""
    _CIT_BRACKET_RX = re.compile(r"\[(?P<num>\d{1,4})\]")
    _CIT_FULL_RX = re.compile(r"【(?P<num>\d{1,4})】")
    _CIT_FOOT_RX = re.compile(r"\[\^(?P<num>\d{1,4})\]")
    TAGS = ("CTX", "PARSED", "DB", "GEN")
    seen = set()
    tag_counts = {t: 0 for t in TAGS}
    for t in texts:
        if not t:
            continue
        for rx in (_CIT_BRACKET_RX, _CIT_FULL_RX, _CIT_FOOT_RX):
            for m in rx.finditer(t):
                try:
                    seen.add(int(m.group("num")))
                except Exception:
                    pass
        for tag in TAGS:
            tag_counts[tag] += len(re.findall(rf"\[{tag}\]", t))
    return {
        "refs": sorted(seen),
        "tags": tag_counts,
        "has_ctx": any(tag_counts[k] > 0 for k in ("CTX", "PARSED", "DB")),
    }


def _split_ctx(ctx: str, max_snips: int = 6, max_tokens: int = 140) -> list[dict]:
    """Split context into short, trimmed blocks for evidence packing."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", ctx or "") if b.strip()]
    snips = []
    for b in blocks:
        t = b
        # crude token limiter ~ chars
        if len(t) > 800:
            t = t[:800] + " …"
        snips.append({"id": None, "source": "CTX", "text": t})
        if len(snips) >= max_snips:
            break
    return snips


def _pick_web_snips(refs: list[dict], max_per_src: int = 1, max_total: int = 6, max_chars: int = 800) -> list[dict]:
    """Extract up to max_total web snippets from references for evidence packing."""
    out = []
    for i, r in enumerate(refs, start=1):
        abs_ = (r.get("abstract") or "").strip()
        if not abs_:
            continue
        txt = abs_
        if len(txt) > max_chars:
            txt = txt[:max_chars] + " …"
        out.append({"id": f"S{i}", "source": "WEB", "text": txt})
    return out[:max_total]


def build_evidence_pack(q: str, vs_ctx: str, refs: list[dict]) -> tuple[str, list[str]]:
    """Builds an EVIDENCE block and id list for strict protocol generation."""
    ctx_snips = _split_ctx(vs_ctx, max_snips=6)
    web_snips = _pick_web_snips(refs, max_total=6)

    lines = ["EVIDENCE"]
    ids: list[str] = []

    # Number web sources S1..Sk and attach one child chunk each (S1_C1)
    for i, r in enumerate(refs, start=1):
        s_id = f"S{i}"
        title = (r.get("title") or "(no title)")[:180]
        year = r.get("year") or ""
        url = r.get("url") or (r.get("doi") and f"https://doi.org/{r['doi']}" ) or ""
        lines.append(f"[{s_id}] {title} ({year}) — {url}".strip())
        abs_ = (r.get("abstract") or "").strip()
        if abs_:
            c_id = f"{s_id}_C1"
            lines.append(f"[{c_id}] \"{abs_[:600]}\"")
            ids.extend([s_id, c_id])

    # Attach CTX snippets as CTX1.. if present
    for j, s in enumerate(ctx_snips, start=1):
        cid = f"CTX{j}"
        lines.append(f"[{cid}] \"{s['text'][:600]}\"")
        ids.append(cid)

    lines.append("END EVIDENCE")
    return "\n".join(lines), ids


def basic_search(query: str, n: int = 6) -> list[dict]:
    """Merge local LOOKUP hits + OpenAlex web hits; de-duplicate by DOI/title."""
    if not query or not query.strip():
        return []
    local = []
    if LOOKUP is not None:
        try:
            hits = LOOKUP.query(query, topk=n)
            if isinstance(hits, pd.DataFrame) and not hits.empty:
                for _, row in hits.fillna("").iterrows():
                    title = row.get("title") or row.get("name") or row.get("__source__") or "table hit"
                    year = row.get("year") or row.get("publication_year") or ""
                    doi = row.get("doi") or ""
                    url = row.get("url") or (f"https://doi.org/{doi}" if doi else "")
                    local.append({
                    "title": _safe_text(title)[:300],
                    "year": _safe_text(year),
                    "url": _safe_text(url),
                    "doi": _safe_text(doi),
                    "source": "lookup",
                    })
        except Exception as e:
            print("[basic_search] lookup query failed:", e)

    web = []
    try:
        web = search_papers(query, n)
    except Exception as e:
        print("[basic_search] OpenAlex fetch failed:", e)

    seen = {(d.get("doi") or d.get("title","")).lower() for d in local}
    for w in web:
        key = (w.get("doi") or w.get("title","")).lower()
        if key not in seen:
            local.append({
                k: w.get(k, "")
                for k in ("title","year","url","doi","abstract","journal","authors")
            } | {"source":"web"})
            seen.add(key)

        print(f"[basic_search] returned {len(local[:2*n])} refs "
            f"(web={sum(1 for r in local if r.get('source')=='web')}, "
            f"lookup={sum(1 for r in local if r.get('source')=='lookup')})")
        
        return local[:2 * n]

# Light stoplist
_STOP = {
    "the","a","an","of","and","or","for","to","in","on","at","by","with","from",
    "using","via","how","what","which","based","study","paper","approach","method",
    "large","scale","facile","simple","novel","new","effect","effects","based","describe",
}

# Common topic vocab (broad, domain-agnostic)
SHAPE_TERMS = {
    "nanorod","nanowire","nanotube","nanofiber","nanoparticle","nanocube","nanoplate",
    "nanosheet","nanosphere","nanocluster","quantum dot","quantum dots","nanobelt",
    "thin film","thin-film","microrod","microfiber","microtube","nanoflake","nanoribbon",
}
PROCESS_TERMS = {
    "hydrothermal","solvothermal","sol-gel","polyol","microwave","autoclave","calcination",
    "anneal","annealing","seed-mediated","galvanic replacement","hot injection","precipitation",
    "co-precipitation","electrodeposition","chemical bath","chemical bath deposition","cbd",
    "cvd","pvd","ald","sputtering","spin coating","dip coating","spray pyrolysis","templated",
    "template-assisted","microemulsion","micelle",
}
MEASURE_TERMS = {
    "xrd","tem","sem","afm","xps","raman","ftir","uv-vis","pl","vsm","squid","fmr","hall",
    "fourier","impedance","conductivity","permittivity","dielectric","magnetization",
}
INTENT_SYN = {
    "synthesize": {"synthesize","synthesis","prepare","fabricate","grow","make","deposit"},
    "characterize": {"characterize","measure","quantify","analyze","determine","evaluate"},
    "theory": {"theory","mechanism","dft","first-principles","simulation","model"},
}

# Prefer reputable publishers
PUBLISHER_BOOST = (
    "acs.org","rsc.org","sciencedirect","elsevier","springer","wiley","nature.com",
    "aip.org","iop.org","electrochem.org","tandfonline","spiedigitallibrary",
    "ieee.org","aps.org","arxiv.org","pnas.org"
)
# Drop obvious non-technical or low-signal hits 
BAD_DOMAINS = (
    "ssrn.com","medium.com","researchgate.net","quora.com","stackexchange",
)

CHEM_FORMULA_RX = re.compile(r"\b(?:[A-Z][a-z]?\d*){1,4}\b")  # light heuristic for formulas

def _norm(x: str) -> str:
    return (x or "").lower()

def _tokenize(q: str) -> List[str]:
    # keep hyphenated, split on non-letters/numbers, drop stopwords
    qn = _norm(q)
    toks = re.findall(r"[a-z0-9][a-z0-9\-]+", qn)
    return [t for t in toks if t not in _STOP and len(t) > 1]

def _extract_formulas(q: str) -> Set[str]:
    return set(CHEM_FORMULA_RX.findall(re.sub(r"\s", "", q)))

def _extract_multiword_phrases(q: str, vocab: Set[str]) -> Set[str]:
    qn = _norm(q)
    found = set()
    for phrase in vocab:
        if phrase in qn:
            found.add(phrase)
    return found


def derive_query_profile(q: str) -> Dict[str, Set[str]]:
    q = q or ""
    # Normalize Unicode subscripts to ASCII before formula detection (Mn₃O₄ -> Mn3O4)
    q_ascii = from_subscript(q)
    toks = set(_tokenize(q_ascii))

    # Materials: formulas + aliases (subscripts, element oxide names, special aliases)
    mats_raw = _extract_formulas(q_ascii)               # e.g., {"Mn3O4"}
    mats = _augment_material_terms(mats_raw)            # {"mn3o4","mn3o₄","manganese oxide","hausmannite",...}

    # Also include simple chemical names typed by user (oxide/nitride/etc.)
    chem_name_hits = set(
        re.findall(
            r"\b([a-z][a-z0-9\-]+(?:\s+(?:oxide|nitride|carbide|sulfide|sulphide|phosphide|ferrite|perovskite|aluminate|titanate|ferrate)))\b",
            _norm(q_ascii)
        )
    )
    mats |= {m.lower() for m in chem_name_hits}

    # Topic tokens
    shapes = _extract_multiword_phrases(q_ascii, SHAPE_TERMS) | (SHAPE_TERMS & toks)
    procs  = _extract_multiword_phrases(q_ascii, PROCESS_TERMS) | (PROCESS_TERMS & toks)
    meas   = _extract_multiword_phrases(q_ascii, MEASURE_TERMS) | (MEASURE_TERMS & toks)

    # Intent inference
    intent = set()
    for k, syns in INTENT_SYN.items():
        if any(s in _norm(q_ascii) for s in syns):
            intent.add(k)
    if not intent and (shapes or procs):
        intent.add("synthesize")

    return {
        "materials": mats,                         # already lower-cased
        "shapes": {s.lower() for s in shapes},
        "processes": {p.lower() for p in procs},
        "measure": {m.lower() for m in meas},
        "intent": intent,
        "tokens": toks,
    }

def _domain_ok(url: str) -> bool:
    u = _norm(url)
    if not u: return True
    return not any(bad in u for bad in BAD_DOMAINS)

def _publisher_score(url: str) -> int:
    u = _norm(url)
    return 1 if any(p in u for p in PUBLISHER_BOOST) else 0

def _blob(r: Dict) -> str:
    return _norm(f"{r.get('title','')} {r.get('abstract','')}")

def _match_any(blob: str, terms: Set[str]) -> int:
    return sum(1 for t in terms if t and t in blob)

def _has_any(blob: str, terms: Set[str]) -> bool:
    return any(t for t in terms if t in blob)

def filter_and_rerank_generic(q: str, refs: List[Dict]) -> List[Dict]:
    """
    General-purpose gate:
      - Prefer matches on materials + (shapes or processes or measure), depending on intent.
      - Drop obvious bad domains.
      - Score by overlap + publisher boost.
      - Adaptive fallback if the query is sparse/broad.
    """
    prof = derive_query_profile(q)
    mats = prof["materials"]
    shapes = prof["shapes"]
    procs  = prof["processes"]
    meas   = prof["measure"]
    intent = prof["intent"]

    candidates = []
    for r in refs or []:
        url = r.get("url") or (r.get("doi") and f"https://doi.org/{r['doi']}") or ""
        if not _domain_ok(url):
            continue
        b = _blob(r)

        # Hard-ish filter: if materials specified, require at least one material token in title/abstract
        if mats and not _has_any(b, mats):
            continue

        # Intent-aware requirement
        want_any_topic = False
        if "synthesize" in intent:
            if shapes or procs:
                if not (_has_any(b, shapes) or _has_any(b, procs)):
                    continue
            else:
                # synthesis intent but no explicit shape/process -> just ensure generic synthesis verbs
                if not any(w in b for w in ("synthesis","prepare","fabricat","grow","deposit","autoclave","hydrothermal","solvothermal","calcination","anneal")):
                    want_any_topic = True  # score but don't drop
        elif "characterize" in intent:
            if meas and not _has_any(b, meas):
                want_any_topic = True
        # theory intent has no strong gate

        # Score
        score = 0
        score += 3 * _match_any(b, mats)         # material is highly important
        score += 2 * _match_any(b, shapes)       # morphology
        score += 2 * _match_any(b, procs)        # process
        score += 1 * _match_any(b, meas)         # measurements
        score += _publisher_score(url)

        # Soft bonus if text contains generic verbs
        if want_any_topic and any(w in b for w in ("synthesis","prepare","fabricat","grow","deposit","measure","characteriz","theory","mechanism","dft")):
            score += 1

        # Keep even low scores to fall back on
        candidates.append((score, r))

    # If materials weren’t specified (truly broad Q), don’t overfilter—just keep top refs by score
    if not mats and not candidates:
        candidates = [(0, r) for r in refs]

    candidates.sort(key=lambda x: x[0], reverse=True)
    filtered = [r for s, r in candidates if s > 0]

    # Adaptive fallbacks
    if len(filtered) < 3:
        # Relax gates: allow matches with any topic terms or generic verbs
        looser = []
        for r in refs or []:
            b = _blob(r)
            score = 0
            score += 2 * _match_any(b, shapes | procs | meas)
            score += 1 * int(any(w in b for w in ("synthesis","prepare","fabricat","grow","deposit","measure","characteriz","dft","theory")))
            score += _publisher_score(r.get("url") or "")
            looser.append((score, r))
        looser.sort(key=lambda x: x[0], reverse=True)
        filtered = [r for s, r in looser[:8] if s > 0] or [r for s, r in candidates[:8]]

    # Cap for evidence pack
    return filtered[:8]

@lru_cache(maxsize=128)
def cached_vs_search(q):
    """Cached vector store search."""
    return vs.search(q, k=8) or ""


@lru_cache(maxsize=128)
def cached_lookup_query(q):
    """Cached lookup query for DuckDB."""
    if LOOKUP is not None:
        try:
            return LOOKUP.query(q, topk=5)
        except Exception:
            return None
    return None


@lru_cache(maxsize=128)
def cached_basic_search(q, n):
    """Cached basic search (local + web)."""
    try:
        return basic_search(q, n) or []
    except Exception:
        return []


# ---- Flask Routes ---- #
@app.get("/health")
def health():
    """Health check endpoint."""
    return "ok", 200


@app.get("/db_health")
def db_health():
    """MongoDB health check endpoint."""
    try:
        mongo_ping()
        return {"mongo": "ok"}, 200
    except Exception as e:
        return {"mongo": "error", "detail": str(e)}, 500


@app.get("/")
def home():
    """Render the home page."""
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return "<h1>NanoChemGPT</h1><p>templates/index.html missing.</p>", 200


# ---- Uploads ---- #
JOBS: dict[str, dict] = {}
def _set_job(jid: str, **kw):
    """Set or update a job status in the JOBS dict."""
    JOBS.setdefault(jid, {}).update(kw)


@app.post("/upload")
def upload():
    """Upload a file and process it for context or table ingestion."""
    f = request.files.get("file")
    if not f or f.filename == "":
        abort(400, "No file uploaded.")
    fname = secure_filename(f.filename or "uploaded_file")
    lower = fname.lower()

    dest = LOOKUP_UPLOAD_DIR / fname if lower.endswith((".parquet", ".csv", ".tsv", ".xlsx")) else (UPLOADS_DIR / fname)
    dest.parent.mkdir(parents=True, exist_ok=True)
    f.save(dest)

    try:
        db = get_db()
        db.uploads.update_one(
            {"filename": fname},
            {"$set": {"filename": fname, "ts": datetime.utcnow(), "status": "received", "path": str(dest)}},
            upsert=True,
        )
    except Exception as e:
        print("[/upload] DB receipt warn:", e)

    jid = os.urandom(8).hex()
    _set_job(jid, status="processing", progress=0, filename=fname)

    try:
        if lower.endswith(".pdf"):
            threading.Thread(target=_process_pdf_job, args=(jid, dest, fname), daemon=True).start()
        elif lower.endswith(".json"):
            txt = dest.read_text(encoding="utf-8", errors="ignore")
            vs.add_to_store(txt, tag=f"upload:{fname}")
            _mark_uploaded(fname, kind="json", status="indexed")
            _set_job(jid, status="done", progress=100)
        elif lower.endswith((".parquet", ".csv", ".tsv", ".xlsx")):
            _mark_uploaded(fname, kind="table", status="stored")
            _set_job(jid, status="done", progress=100)
        else:
            txt = dest.read_text(encoding="utf-8", errors="ignore")
            vs.add_to_store(txt, tag=f"upload:{fname}")
            _mark_uploaded(fname, kind="text", status="indexed")
            _set_job(jid, status="done", progress=100)
    except Exception as e:
        _set_job(jid, status="error", error=str(e))

    return jsonify({"ok": True, "job_id": jid, "filename": fname, "path": str(dest)})


def _mark_uploaded(fname: str, *, kind: str, status: str):
    """Mark a file as uploaded and update DB status."""
    try:
        get_db().uploads.update_one(
            {"filename": fname},
            {"$set": {"status": status, "indexed_at": datetime.utcnow(), "kind": kind}},
            upsert=True,
        )
    except Exception as e:
        print("[/upload] DB update warn:", e)


@app.get("/status/<jid>")
def status(jid: str):
    """Get the status of an upload or processing job."""
    j = JOBS.get(jid)
    if not j:
        abort(404, "unknown job id")
    return jsonify(j)


def _process_pdf_job(jid: str, path: Path, filename: str):
    """Extract text from PDF and add to vector store."""
    from PyPDF2 import PdfReader
    db = None
    try:
        try:
            db = get_db()
        except Exception as e:
            print("[/upload] get_db failed (continuing without DB):", e)
        reader = PdfReader(str(path))
        n = len(reader.pages) or 1
        texts = []
        for i, page in enumerate(reader.pages, 1):
            texts.append(page.extract_text() or "")
            _set_job(jid, progress=int(100 * i / n))
        text = "\n".join(texts)
        if not text or not text.strip():
            raise ValueError("PDF contains no extractable text.")
        vs.add_to_store(text, tag=f"upload:{filename}")
        if db is not None:
            db.uploads.update_one(
                {"filename": filename},
                {"$set": {"status": "indexed", "indexed_at": datetime.utcnow(), "n_pages": n}},
                upsert=True,
            )
        _set_job(jid, status="done", progress=100)
    except Exception as e:
        if db is not None:
            try:
                db.uploads.update_one(
                    {"filename": filename},
                    {"$set": {"status": "error", "error": str(e), "indexed_at": datetime.utcnow()}},
                    upsert=True,
                )
            except Exception:
                pass
        _set_job(jid, status="error", error=str(e))


# ---- Search API ---- #
@app.post("/search")
def search_route():
    """Search for references using local and web sources."""
    payload = request.get_json(silent=True) or {}
    q = (payload.get("q") or payload.get("query") or "").strip()
    n = int(payload.get("n") or 6)
    if not q:
        abort(400, "Missing 'q' (query).")
    refs = basic_search(q, n)
    return jsonify({"results": refs})


def _validate_strict_json(raw_json: str, evidence_ids: list[str]) -> str:
    """Ensure every sentence in procedure[].text and reasoning[].claim has a bracketed id that exists."""
    try:
        data = json.loads(raw_json)
    except Exception:
        # ask model to fix formatting
        fix = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role":"system","content":"Fix the JSON to be valid UTF-8 JSON; do not change semantics."},
                      {"role":"user","content":raw_json}]
        ).choices[0].message.content
        if fix is None:
            raise ValueError("Unable to fix JSON")
        data = json.loads(fix)

    def has_valid_cite(s: str) -> bool:
        ids = re.findall(r"\[([A-Z0-9_]+)\]", s or "")
        return any(i in evidence_ids for i in ids)

    data.setdefault("uncited", [])
    data.setdefault("insufficient_evidence", [])

    # Validate procedure steps
    proc_out = []
    for step in data.get("procedure", []):
        txt = (step.get("text") or "").strip()
        if not has_valid_cite(txt):
            data["uncited"].append(txt)
        else:
            proc_out.append(step)
    data["procedure"] = proc_out

    # Validate reasoning claims
    reas_out = []
    for cl in data.get("reasoning", []):
        claim = (cl.get("claim") or "").strip()
        cits = cl.get("citations") or []
        if not claim or not any(c in evidence_ids for c in cits):
            data["uncited"].append(claim)
        else:
            reas_out.append(cl)
    data["reasoning"] = reas_out

    return json.dumps(data, ensure_ascii=False, indent=None)


# ---- Ask API ---- #
@app.post("/ask")
def ask():
    """Main endpoint for question answering and protocol generation."""
    def split_reasoning(raw: str) -> tuple[str, str]:
        """Split answer and rationale from model output."""
        if not raw:
            return "", ""
        text = raw.strip()
        fence_rx = re.compile(r"```(?:reason|rationale|reasoning)\s*([\s\S]*?)```", re.I)
        rationale = ""
        fences = list(fence_rx.finditer(text))
        if fences:
            rationale = fences[-1].group(1).strip()
            answer = fence_rx.sub("", text).strip()
            return answer, rationale
        head = re.compile(r"(?:^|\n)#{1,3}\s*(rationale|reasoning)\b[^\n]*\n((?:.*\n?)*)$", re.I | re.S)
        m = head.search(text)
        if m:
            rationale = m.group(2).strip()
            answer = text[:m.start()].strip()
            return answer, rationale
        return text, ""

    try:
        payload = request.get_json(silent=True) or {}
        q = (payload.get("question") or "").strip()
        mode = (payload.get("mode") or "robot").strip().lower()
        want_inline = bool(payload.get("want_inline_citations", True))
        if not q:
            abort(400, "No question.")

        # vector context
        vs_ctx = ""
        try:
            vs_ctx = vs.search(q, k=8) or ""
        except Exception as e:
            print("[/ask] vs.search error:", e)

        # table lookup context
        table_ctx = ""
        table_refs = []
        if LOOKUP is not None:
            try:
                hits = LOOKUP.query(q, topk=5)
                rows = hits.to_dict(orient="records")
                lines = []
                for i, row in enumerate(rows, start=1):
                    solvent = row.get("solvent") or row.get("solvent_system")
                    temp = row.get("temp_C") or row.get("temperature_C")
                    time_h = row.get("time_h") or row.get("duration_h")
                    note = row.get("notes") or ""
                    line = f"[T{i}] solvent={solvent}; temp_C={temp}; time_h={time_h}; {note}".strip()
                    lines.append(line)
                    url = row.get("url") or (row.get("doi") and f"https://doi.org/{row['doi']}")
                    if url:
                        table_refs.append({"title": f"Table row {i}", "url": url})
                table_ctx = "\n".join(lines)
            except Exception as e:
                print("[/ask] LOOKUP query error:", e)

        # join contexts
        context_parts = []
        if vs_ctx:
            context_parts.append("<<<CTX_UPLOADS>>>\n" + (vs_ctx if isinstance(vs_ctx, str) else str(vs_ctx)))
        if table_ctx:
            context_parts.append("<<<CTX_TABLE>>>\n" + (table_ctx if isinstance(table_ctx, str) else str(table_ctx)))
        context_joined = "\n\n---\n\n".join(context_parts).strip()

        refs = []
        try:
            refs = basic_search(q, n=20) or []
        except Exception as e:
            print("[/ask] basic_search error:", e)
        if table_refs:
            refs = list(refs) + table_refs

        # General relevance filter + reranker 
        refs_filtered = filter_and_rerank_generic(q, refs)
        if refs_filtered:
            refs = refs_filtered

        # rebuild refs_prompt after possibly changing refs
        def _ref_url(r):
            if r.get("url"): return r["url"]
            if r.get("doi"): return f"https://doi.org/{r['doi']}"
            return ""
        refs_prompt = "\n".join(
            f"[{i+1}] {(r.get('title') or '(no title)')} ({r.get('year') or ''}) — {_ref_url(r)}"
            for i, r in enumerate(refs)
        ).strip()

        print("[/ask] refs:")
        for i, r in enumerate(refs, 1):
            print(f"  [{i}] {r.get('title')} — {r.get('doi') or r.get('url')}")

        if not refs:
            return jsonify({
                "answer": "",
                "rationale": "",
                "references": [],
                "error": "No sufficiently relevant references found for this query; please rephrase or provide seed refs."
            }), 422

        # ========== STRICT PATH ==========
        if mode in ("robot_strict", "reasoning_strict"):
            evidence_text, evidence_ids = build_evidence_pack(q, context_joined, refs)

            STRICT_SCHEMA = (
                'You are writing for materials chemists. Use ONLY the passages in EVIDENCE.\n'
                'If a claim or step is not supported there, list it under "insufficient_evidence".\n'
                'Every sentence that makes a claim MUST include bracket citations to [S#], [S#_C#] or [CTX#].\n'
                'Do NOT use prior knowledge.\n'
                'Return valid JSON only, matching this schema exactly:\n'
                '{\n'
                ' "procedure": [ {"step": 1, "text": "… [S1_C1][CTX1]"}, ... ],\n'
                ' "reasoning": [ {"claim": "…", "citations": ["S1_C1","CTX1"]}, ... ],\n'
                ' "uncited": [],\n'
                ' "insufficient_evidence": []\n'
                '}\n'
            )

            strict_task = (
                f'TASK\n'
                f'Write a step-by-step procedure and a brief reasoning section for: "{q}".\n'
                f'{evidence_text}\n'
                f'END EVIDENCE\n'
                f'OUTPUT JSON ONLY.'
            )

            raw = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": STRICT_SCHEMA},
                    {"role": "user", "content": strict_task},
                ],
            ).choices[0].message.content

            answer = _validate_strict_json((raw or "").strip(), evidence_ids)
            rationale = ""  # reasoning lives inside the JSON
            used_summary = {"refs": [], "tags": {}, "has_ctx": bool(context_joined)}

        # ========== LEGACY PATHS ==========
        else:
            robot_rules = (
                "Return a discrete lab protocol with exact quantities on a small scale (~0.5 mmol Co):\n"
                " - Include specific masses (mg) or mmol for reagents; volumes (mL) for liquids.\n"
                " - Specify temperatures (°C), ramp rates (°C/min), hold times (min/h), and atmosphere (Ar/N2/vacuum).\n"
                " - Include workup and purification (quench, washing/centrifugation, drying) with volumes.\n"
                " - No placeholders (avoid “e.g.”/“or”). Be decisive.\n"
                " - Output only the final protocol in markdown. Do not include any fenced blocks named reason or rationale in the answer. Put all reasoning in the separate rationale channel."
            )
            reasoning_rules = (
                " - Provide a mechanistic explanation and design considerations for the target.\n"
                " - Focus on: nucleation vs growth; ligand/solvent coordination; surfactants; "
                " - reduction/oxidation; temperature profile and morphology control; atmosphere; pitfalls; safety.\n"
                " - Do NOT return a step-by-step protocol. Be concise but specific."
            )
            inline_rule = (
                " - When you pull a fact from any numbered REFERENCE, put its number in square brackets right after the sentence "
                "(e.g. “hydrothermal at 200 °C [3]”)." if want_inline else
                " - Inline numeric citations are optional for this request."
            )
            acs_rule = (
                " - Write the REFERENCES block in ACS format: author(s), title, journal, year, volume, pages, DOI.\n"
                " - Use inline numeric citations ([n]) for facts from REFERENCES. Do NOT include a REFERENCES block in your answer."
            )

            def strip_references_block(text: str) -> str:
                return re.sub(r"## References[\s\S]*", "", text, flags=re.I).strip()

            if mode == "reasoning":
                prompt = (
                    "You are NanoChemGPT. Use the CONTEXT and numbered REFERENCES.\n"
                    "Rules:\n"
                    " - Prefer CONTEXT and REFERENCES over general knowledge when relevant.\n"
                    " - For each bullet, quote or paraphrase a specific finding from CONTEXT or REFERENCES, and cite the source. Do not generalize or invent citations.\n"
                    " - If you use any content from CONTEXT, append [CTX] on that line.\n"
                    f"{inline_rule}\n"
                    " - If CONTEXT is insufficient, say so explicitly before generalizing.\n"
                    " - For each cited reference, briefly summarize the relevant finding and explain how it relates to aspect ratio and temperature.\n"
                    " - If no reference supports a statement, say so explicitly and do not cite it.\n"
                    f"{reasoning_rules}\n"
                    f"{acs_rule}\n"
                    "Return exactly ONE block:\n"
                    "## Mechanistic reasoning\n"
                    "- bullet points with inline [n] and [CTX] where appropriate.\n\n"
                    f"CONTEXT:\n{context_joined}\n\n"
                    f"REFERENCES:\n{refs_prompt}\n\n"
                    f"User question: {q}"
                )
            else:
                prompt = (
                    "You are NanoChemGPT. Use the CONTEXT and the numbered REFERENCES to propose a synthesis.\n"
                    "Rules:\n"
                    " - Prefer CONTEXT and REFERENCES over general knowledge when relevant.\n"
                    " - For each step, quote or paraphrase a specific finding from CONTEXT or REFERENCES, and cite the source. Do not generalize or invent citations.\n"
                    " - If you use any content from CONTEXT, append [CTX] on that line.\n"
                    f"{inline_rule}\n"
                    " - If CONTEXT is insufficient, say so explicitly before generalizing.\n"
                    f"{robot_rules}\n"
                    f"{acs_rule}\n"
                    "Return two blocks exactly in this order:\n"
                    "## Synthesis Protocol:\n"
                    "1. **Hardware & Glassware**:\n[]\n"
                    "2. **Materials**:\n[]\n"
                    "3. **Procedure**\n[]\n\n"
                    "```reason\n"
                    "For each key justification, add inline tags: [CTX] for uploaded/context hits, [DB] for Mongo Q&A, "
                    "[PARSED] for parsed protocols, [n] for numbered web REFERENCES, [GEN] if inferred.\n"
                    "Keep rationales terse.\n"
                    "Add NO other blocks of text.\n"
                    "```\n\n"
                    f"CONTEXT:\n{context_joined}\n\n"
                    f"REFERENCES:\n{refs_prompt}\n\n"
                    f"User question: {q}"
                )

            raw = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            ).choices[0].message.content

            # Split answer/rationale
            if mode == "reasoning":
                answer = strip_references_block((raw or "").strip())
                rationale = ""
            else:
                answer, rationale = split_reasoning(strip_references_block(raw or ""))

            # Rationale fallback if missing
            if not (rationale or "").strip():
                try:
                    rationale_only = (
                        "You previously produced the SynthesisProtocol below.\n"
                        "Write a short rationale (5–8 bullets max). For each key justification add inline tags:\n"
                        "[CTX] uploaded/context hits, [PARSED] parsed protocols, [n] for numbered web REFERENCES, [GEN] for general.\n"
                        "Return just the rationale text, no code fences, no extra headings."
                    )
                    rraw = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": rationale_only},
                            {"role": "user", "content": f"CONTEXT:\n{context_joined}"},
                            {"role": "user", "content": f"REFERENCES:\n{refs_prompt}"},
                            {"role": "user", "content": f"ANSWER:\n{(answer or '').strip()}"},
                            {"role": "user", "content": f"QUESTION:\n{q}"},
                        ],
                        temperature=0.2,
                    ).choices[0].message.content
                    rationale = (rraw or "").strip()
                except Exception as e:
                    print("[/ask] rationale fallback failed:", e)
                    rationale = rationale or ""

            # Post-pass: enforce citations & CTX usage if missing (legacy only)
            try:
                used_summary = _extract_used_markers(answer or "", rationale or "")
                if want_inline and not used_summary.get("refs"):
                    try:
                        revise_refs = client.chat.completions.create(
                            model="gpt-4o-mini",
                            temperature=0,
                            messages=[
                                {"role": "system", "content": (
                                    "Add inline [n] citations wherever information was taken from the numbered REFERENCES list. "
                                    "Do NOT remove any existing [CTX] content. Only insert citations where appropriate."
                                )},
                                {"role": "user", "content": f"REFERENCES:\n{refs_prompt}"},
                                {"role": "user", "content": f"ORIGINAL ANSWER:\n{answer}"}
                            ]
                        ).choices[0].message.content
                        if revise_refs and len(revise_refs) >= 0.7 * len(answer):
                            answer = revise_refs
                            used_summary = _extract_used_markers(answer, rationale)
                    except Exception as e:
                        print("[ask] ref-revise step failed:", e)
            except Exception as _e:
                print("[/ask] used extraction failed:", _e)
                used_summary = {"refs": [], "tags": {}, "has_ctx": False}

            if context_joined and not used_summary.get("has_ctx"):
                try:
                    revise = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "Revise the answer to explicitly use CONTEXT where relevant. Insert [CTX] markers on lines that derive from CONTEXT, and prefer CONTEXT over general knowledge. Do not change structure."},
                            {"role": "user", "content": f"CONTEXT:\n{context_joined}"},
                            {"role": "user", "content": f"ORIGINAL ANSWER:\n{answer}"}
                        ]
                    ).choices[0].message.content
                    if revise and len(revise) >= 0.7 * len(answer):
                        answer = revise
                        used_summary = _extract_used_markers(answer, rationale or "")
                except Exception as e:
                    print("[ask] revise step skipped:", e)

        # ========== COMMON DB INSERT + RETURN (both paths) ==========
        qa_id = None
        try:
            db = get_db()
            ins = db.qa.insert_one({
                "created_at": datetime.utcnow(),
                "question": q,
                "mode": mode,
                "answer": (answer or "").strip(),
                "rationale": rationale,
                "references": refs,
                "refs_used": used_summary.get("refs", []),
                "used_tags": used_summary.get("tags", {}),
                "ctx_vs": vs_ctx,
                "ctx_table": table_ctx,
            })
            qa_id = str(ins.inserted_id)
        except Exception as e:
            print("[/ask] DB insert warn:", e)

        return jsonify({
            "answer": (answer or "").strip(),
            "rationale": rationale,
            "references": refs,
            "refs": refs,
            "refs_used": used_summary.get("refs", []),
            "used": used_summary,
            "mode": mode,
            "qa_id": qa_id,
            "ctx_vs": (vs_ctx if isinstance(vs_ctx, str) else str(vs_ctx or ""))[:8000],
            "ctx_table": (table_ctx if isinstance(table_ctx, str) else str(table_ctx or ""))[:4000],
        })

    except Exception as e:
        print("[/ask] Unhandled error:", e)
        traceback.print_exc()
        return jsonify({"error": f"/ask failed: {e}"}), 500

# ---- Parse & Save ---- #
@app.post("/parse")
def parse_route():
    try:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "JSON must contain non-empty 'text'"}), 400
        data = convert_text_to_robot_ops(text)
        return jsonify({"ok": True, "data": data})
    except ValueError as ve:
        return jsonify({"ok": False, "error": str(ve)}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"parse failed: {e}"}), 500

@app.post("/save_txt")
def save_txt():
    data = request.get_json(silent=True) or {}
    answer = (data.get("answer") or "").strip()
    question = (data.get("question") or "").strip()
    if not answer:
        abort(400, "answer is empty")
    buf = io.BytesIO(f"Q: {question}\n\nA:\n{answer}\n".encode())
    buf.seek(0)
    fname = f"chatau_{datetime.utcnow():%Y%m%d_%H%M%S}.txt"
    return send_file(buf, mimetype="text/plain", as_attachment=True, download_name=fname)

@app.post("/parse_upload")
def parse_upload():
    try:
        f = request.files.get("file")
        if not f or f.filename == "":
            return jsonify({"ok": False, "error": "no file"}), 400
        try:
            text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            text = f.read()
        if not (text or "").strip():
            return jsonify({"ok": False, "error": "file is empty"}), 400
        data = validate_step(text)
        return jsonify({"ok": True, "data": data})
    except ValueError as ve:
        return jsonify({"ok": False, "error": str(ve)}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"parse_upload failed: {e}"}), 500

@app.post("/clear_uploads")
def clear_uploads_route():
    try:
        vs.clear_uploads()
    except Exception as e:
        print("clear_uploads error:", e)
    return {"status": "uploads cleared"}

def _safe_id(x):
    try:
        from bson import ObjectId
        return ObjectId(x)
    except Exception:
        return None

def _doc(obj):
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    for k, v in list(out.items()):
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out

@app.get("/api/history")
def api_history():
    db = get_db()
    try:
        skip = int(request.args.get("skip", 0))
        limit = min(100, int(request.args.get("limit", 10)))
    except Exception:
        skip, limit = 0, 10
    q = (request.args.get("q") or "").strip()
    cur = None
    if q:
        try:
            cur = db.qa.find({"$text": {"$search": q}})
        except Exception:
            cur = db.qa.find({"question": {"$regex": q, "$options": "i"}})
    else:
        cur = db.qa.find({})
    items = [_doc(d) for d in cur.sort("created_at", -1).skip(skip).limit(limit)]
    return jsonify({"items": items, "skip": skip, "limit": limit})

@app.get("/api/history/<id>")
def api_history_one(id):
    db = get_db()
    oid = _safe_id(id)
    if not oid:
        abort(404, "invalid id")
    doc = db.qa.find_one({"_id": oid})
    if not doc:
        abort(404, "not found")
    return jsonify(_doc(doc))

@app.get("/api/uploads")
def api_uploads():
    db = get_db()
    try:
        limit = min(200, int(request.args.get("limit", 50)))
    except Exception:
        limit = 50
    cur = db.uploads.find({}).sort([("indexed_at", -1), ("ts", -1)]).limit(limit)
    items = [_doc(d) for d in cur]
    return jsonify({"items": items, "limit": limit})

@app.post("/admin/rebuild_mech_index")
def rebuild_mech_index():
    from retriever.retriever import build_index, Embedder
    idx, meta = build_index(Embedder())
    return {"ok": True, "entries": len(meta)}

@app.errorhandler(400)
@app.errorhandler(422)
@app.errorhandler(500)
def handle_err(e):
    return jsonify(error=str(e)), getattr(e, "code", 500)

@app.errorhandler(413)
def too_large(e):
    return jsonify(error="File bigger than 100 MB — compress or split it."), 413

@app.post("/upload_builtin")
def upload_builtin():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"ok": False, "error": "No files uploaded"}), 400
    saved = []
    for f in files:
        fname = secure_filename(f.filename or "uploaded_file")
        dest = BUILTIN_DIR / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
        saved.append(fname)
    return jsonify({"ok": True, "files": saved})


# ---- Main Entrypoint ---- #
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        debug=os.getenv("DEBUG", "0") == "1"
    )
