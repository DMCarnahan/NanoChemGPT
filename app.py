"""
app.py
-----
Main Flask application for NanoChemGPT.
Handles API endpoints, configuration, and integration with search, database, and LLM services.
"""

from __future__ import annotations

# Standard library imports
import os
import io
import re
import glob
import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from functools import lru_cache
import difflib as _difflib
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Third-party imports
import httpx
from flask import Flask, request, jsonify, abort, render_template, send_file, g
from jinja2 import TemplateNotFound
from openai import OpenAI
from werkzeug.utils import secure_filename

# Local modules
import vector_store as vs
from vector_store.uploads_vector import UploadsVectorSearch
from converter import validate_step, convert_text_to_robot_ops
from mongo_client import get_db
from decider.miner_queue import enqueue_text_mining_job
from DuckDB.duck_searcher import get_duck_searcher
from ref_utils import (
    dedupe_and_rerank, extract_used_ref_indexes, renumber_citations,
    split_used_refs, DEFAULT_NANOCHEM_TERMS, format_references_block
)
try:
    from app_utils.helpers import (
        classify_intent as _h_classify_intent,
        kb_search as _h_kb_search,
        kb_fetch as _h_kb_fetch,
        judge_sufficiency as _h_judge_sufficiency,
        _safe_text as _h_safe_text,
        env_int as _env_int,
        env_float as _env_float,
        judge_hits, Hit as _Hit, renumber_citations as _h_renumber_citations,
    )
except Exception:
    _h_classify_intent = _h_kb_search = _h_kb_fetch = None
    _h_judge_sufficiency = _h_safe_text = _h_extract_used_ref_indexes = None
    _h_renumber_citations = None
    _Hit = None
    # --- Safe fallbacks so /ask never 500s if helpers are missing or envs are blank ---
    def _env_int(name: str, default: int) -> int:
        v = os.getenv(name, "")
        v = "" if v is None else v.strip()
        try:
            return default if v == "" else int(v)
        except Exception:
            return default

    def _env_float(name: str, default: float) -> float:
        v = os.getenv(name, "")
        v = "" if v is None else v.strip()
        try:
            return default if v == "" else float(v)
        except Exception:
            return default

# -------------------- Paths/Config --------------------
ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"
BUILTIN_DIR = Path(os.getenv("BUILTIN_DIR", ROOT / "builtin")).resolve()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "/mnt/data/uploads")).resolve()
ATTACH_DIR = Path(os.environ.get('ATTACH_DIR', '/mnt/data/attachments'))
ATTACH_DIR.mkdir(parents=True, exist_ok=True)
LOOKUP_UPLOAD_DIR = Path(os.getenv("LOOKUP_UPLOAD_DIR", "/mnt/data/datasets")).resolve()
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "/mnt/data/index")).resolve()

BUNDLE_AUTO   = ROOT / "harvester" / "out_auto" / "bundle.jsonl"
BUNDLE_MERGED = ROOT / "harvester" / "out_auto" / "bundle_with_methods.jsonl"
INDEX_DIR     = ROOT / "retriever" / "index"

# Ensure required directories exist
for d in (BUILTIN_DIR, UPLOADS_DIR, LOOKUP_UPLOAD_DIR, VECTORSTORE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Flask app setup --------------------
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.config["JSON_AS_ASCII"] = False  # allow UTF-8

# No CSRF protection needed for this application
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "change-me")

# Ephemeral per-question attachments
ATTACH_DIR = Path(os.environ.get("ATTACH_DIR", "/mnt/data/attachments"))
ATTACH_DIR.mkdir(parents=True, exist_ok=True)

def _extract_pdf_text(path: Path, max_pages: int = 40) -> tuple[str, int]:
    try:
        try:
            from pypdf import PdfReader as _PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader as _PdfReader
            except Exception:
                _PdfReader = None
        if _PdfReader is None:
            raise ImportError("pypdf/PyPDF2 not installed")
        reader = _PdfReader(str(path))
        pages = getattr(reader, "pages", [])
        out = []
        for i, page in enumerate(pages, 1):
            if i > max_pages: break
            try:
                out.append(page.extract_text() or "")
            except Exception:
                out.append("")
        return ("\n".join(out), len(pages) or len(out))
    except Exception as e:
        try: app.logger.warning(f"[_extract_pdf_text] {path.name}: {e}")
        except Exception: pass
        return ("", 0)

def _best_chunks_from_text(text: str, query: str, max_chunk_chars: int = 1200, top_k: int = 3):
    import re as _re
    if not text: return []
    q = {t for t in _re.findall(r"[A-Za-z0-9]{3,}", (query or "").lower())} or set(_re.findall(r"[A-Za-z0-9]{3,}", text.lower()))
    chunks, buf, size = [], [], 0
    for para in text.splitlines():
        if size + len(para) + 1 > max_chunk_chars and buf:
            chunks.append("\n".join(buf)); buf, size = [], 0
        buf.append(para); size += len(para) + 1
    if buf: chunks.append("\n".join(buf))
    def score(s: str): 
        toks = set(_re.findall(r"[A-Za-z0-9]{3,}", s.lower()))
        return sum(1 for t in toks if t in q)
    return sorted(chunks, key=score, reverse=True)[:top_k]

@app.before_request
def _inject_base_path():
    # request.script_root will be "/app" when mounted at /app, else ""
    g.base_path = request.script_root or ""

# ──────────────── DuckDB setup ──────────────── #

def maybe_build_duckdb():
    """
    Create a .duckdb from Parquet once if DUCKDB_BOOTSTRAP=1 and files exist.
    Initializes the DuckDB database for tabular data search.
    """
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
        result = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
        rows = result[0] if result else 0
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

# ──────────────── OpenAI client ──────────────── #
_no_proxy = httpx.Client(trust_env=False, timeout=120.0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY, http_client=_no_proxy) if OPENAI_API_KEY else None

RETRIEVER_URL = os.getenv("RETRIEVER_URL", f"http://localhost:{os.getenv('PORT','8000')}/retriever")

def retriever_search(query: str, k: int = 8, level: str|None = None,
                    k_doc: int|None = None, k_passage: int|None = None,
                    w_doc: float|None = None, w_passage: float|None = None,
                    **_: dict) -> list[dict]:
    try:
        payload = {"query": query, "k": int(k)}
        if level: payload["level"] = str(level)
        if k_doc is not None: payload["k_doc"] = int(k_doc)
        if k_passage is not None: payload["k_passage"] = int(k_passage)
        if w_doc is not None: payload["w_doc"] = float(w_doc)
        if w_passage is not None: payload["w_passage"] = float(w_passage)
        r = _no_proxy.post(f"{RETRIEVER_URL.rstrip('/')}/search", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("hits", []) if isinstance(data, dict) else []
    except Exception as e:
        app.logger.warning(f"[retriever] search failed: {e}")
        return []

# ──────────────── Utilities ──────────────── #
def _s(x):
    # Hardened sanitizer prefers helpers._safe_text
    try:
        from app_utils.helpers import _safe_text as _h_safe_text  # local import to avoid circulars
        return _h_safe_text(x)
    except Exception:
        return str(x).strip() if x is not None else ""

def _safe_id(x):
    try:
        from bson import ObjectId
        return ObjectId(x)
    except Exception:
        return None
    
def _stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(x) for x in obj]
    return obj

def _doc(obj):
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    for k, v in out.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out

# ---- Citation extraction and formatting helpers ----
def _extract_used_markers(*texts: str) -> dict:
    """Find [n] citations and [CTX]/[PARSED]/[DB]/[GEN] tags."""
    _CIT_BRACKET_RX = re.compile(r"\[(?P<num>\d{1,4})\]")
    _CIT_FULL_RX = re.compile(r"【(?P<num>\d{1,4})】")
    _CIT_FOOT_RX = re.compile(r"\[\^(?P<num>\d{1,4})\]")
    TAGS = ("CTX", "PARSED", "DB", "GEN")
    seen = set()
    tag_counts = {t: 0 for t in TAGS}
    for t in texts:
        if not t: continue
        for rx in (_CIT_BRACKET_RX, _CIT_FULL_RX, _CIT_FOOT_RX):
            for m in rx.finditer(t):
                try: seen.add(int(m.group("num")))
                except Exception: pass
        for tag in TAGS:
            tag_counts[tag] += len(re.findall(rf"\[{tag}\]", t))
    return {"refs": sorted(seen), "tags": tag_counts, "has_ctx": any(tag_counts[k] > 0 for k in ("CTX", "PARSED", "DB"))}

def build_references_payload(
    answer_text: str,
    refs_input: list[dict],
    *,
    question: str = "",
    top_k: int = 40
) -> dict:
    """
    Build a consistent references payload using ref_utils:
    - dedupe + rerank
    - extract used refs
    - produce refs_all/refs_used/index_map/candidates
    """
    try:
        refs_all = dedupe_and_rerank(
            question or "",
            refs_input or [],
            domain_terms=DEFAULT_NANOCHEM_TERMS,
            top_k=max(top_k, len(refs_input or []))
        )
    except Exception:
        refs_all = list(refs_input or [])

    try:
        used = extract_used_ref_indexes(answer_text or "")
    except Exception:
        used = []

    try:
        refs_used, index_map = split_used_refs(refs_all, used)
    except Exception:
        refs_used, index_map = list(refs_all), {i+1: i+1 for i in range(len(refs_all))}

    return {
        "refs_all": refs_all,
        "refs_used": refs_used,
        "index_map": index_map,
        "candidates": refs_all,
    }

@lru_cache(maxsize=128)
def cached_vs_search(q):
    return vs.search(q, k=8) or ""

@lru_cache(maxsize=128)
def cached_lookup_query(q):
    if LOOKUP is not None:
        try:
            return LOOKUP.query(q, topk=5)
        except Exception:
            return None
    return None

# ──────────────── Routes ──────────────── #
@app.get("/health")
def health():
    return "ok", 200

@app.get("/db_health")
def db_health():
    try:
        db = get_db()
        db.command("ping")
        return {"mongo": "ok"}, 200
    except Exception as e:
        return {"mongo": "error", "detail": str(e)}, 500

@app.get("/")
def home():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return "<h1>NanoChemGPT</h1><p>templates/index.html missing.</p>", 200

# ---- Uploads ---- #
JOBS: dict[str, dict] = {}
def _set_job(jid: str, **kw): JOBS.setdefault(jid, {}).update(kw)

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        abort(400, "No file uploaded.")
    fname = secure_filename(f.filename or "")
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
        app.logger.warning(f"[/upload] DB receipt warn: {e}")

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
    try:
        get_db().uploads.update_one(
            {"filename": fname},
            {"$set": {"status": status, "indexed_at": datetime.utcnow(), "kind": kind}},
            upsert=True,
        )
    except Exception as e:
        app.logger.warning(f"[/upload] DB update warn: {e}")

@app.get("/status/<jid>")
def status(jid: str):
    j = JOBS.get(jid)
    if not j:
        abort(404, "unknown job id")
    return jsonify(j)

def _process_pdf_job(jid: str, path: Path, filename: str):
    from PyPDF2 import PdfReader
    db = None
    try:
        try:
            db = get_db()
        except Exception as e:
            app.logger.warning(f"[/upload] get_db failed (continuing without DB): {e}")
        reader = PdfReader(str(path))
        n = len(reader.pages) or 1
        texts = []
        for i, page in enumerate(reader.pages, 1):
            texts.append(page.extract_text() or "")
            _set_job(jid, progress=int(100 * i / n))
        text = "\n".join(texts)
        if not text.strip():
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

# ---- Ask ---- #
@app.post("/ask")
def ask():
    """
    Unified Q&A endpoint that:
      • Classifies intent (classify_intent) to steer behavior (mode, search breadth).
      • Pulls context from uploads, DuckDB, and **KB** (kb_search/kb_fetch).
      • Builds a numbered REFERENCES list (web + KB), asks the LLM to cite with [n].
      • Extracts **used** citation indexes and returns only those in an ACS-style block.
      • Computes usage markers via _extract_used_markers.
      • Judges sufficiency and, if thin, optionally harvests more data, reindexes, reloads retriever, and retries once.
    """

    # ---------- request payload ----------
    answer = ""
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or payload.get("q") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400
    enqueued = False
    MIN_HITS  = _env_int("JUDGE_MIN_HITS", 1)
    MIN_SCORE = _env_float("JUDGE_MIN_SCORE", 0.15)
    MIN_CHARS = _env_int("JUDGE_MIN_CHARS", 64)

    # ---------- intent classification ----------
    try:
        # Prefer decider if available
        from decider.intent import classify_intent as _decide_intent
        ci = _decide_intent(question)
    except Exception as e:
        # Fallback to helper’s heuristic
        ci = _h_classify_intent(question) if _h_classify_intent else "reason"

    # Normalize classify_intent output to a dict
    if isinstance(ci, str):
        ci = {"intent": ci}
    elif not isinstance(ci, dict):
        ci = {}

    # helpers to coerce types from payload/ci
    def _coerce_bool(v):
        if isinstance(v, bool): return v
        if v is None: return None
        if isinstance(v, (int, float)): return bool(v)
        if isinstance(v, str):
            t = v.strip().lower()
            if t in {"1","true","yes","y","on"}:  return True
            if t in {"0","false","no","n","off"}: return False
        return None

    def _pick_bool(key, default):
        v = payload.get(key, None)
        b = _coerce_bool(v) if v is not None else None
        if b is None:
            b = _coerce_bool(ci.get(key)) if isinstance(ci, dict) else None
        return default if b is None else b

    def _pick_int(key, default):
        for source in (payload, ci):
            if isinstance(source, dict) and key in source:
                try:
                    return int(source[key])
                except Exception:
                    pass
        return default

    # final intent/mode + knobs
    intent = payload.get("intent") or ci.get("intent") or "protocol"
    mode   = payload.get("mode")   or ci.get("mode")
    if not mode:
        mode = "reasoning" if intent in {"reasoning", "analysis"} else "protocol"

    want_inline = _pick_bool("want_inline", True)
    allow_fetch = _pick_bool("allow_fetch", True)
    kb_k        = _pick_int("kb_k", 5)
    web_k       = _pick_int("web_k", 10)

    # Choose retriever level and knobs
    retriever_level = (payload.get("retrieval") or payload.get("retriever_level") or
                       os.getenv("RETRIEVER_LEVEL_DEFAULT") or
                       ("both" if (mode == "protocol" or intent in {"protocol","synthesis","methods"}) else "doc"))
    k_doc_default = min(web_k, 6)
    k_pass_default = max(web_k, 10)
    k_doc = int(payload.get("k_doc", k_doc_default))
    k_passage = int(payload.get("k_passage", k_pass_default))
    w_doc = float(payload.get("w_doc", os.getenv("WEIGHT_DOC", 0.6)))
    w_passage = float(payload.get("w_passage", os.getenv("WEIGHT_PASSAGE", 0.4)))

        # ----------------- uploads → semantic context -----------------
    uploads_ctx = ""
    try:
        uploads_dir = UPLOADS_DIR
        uploads_dir.mkdir(exist_ok=True)
        try:
            import torch
            vector_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            vector_device = "cpu"

        try:
            uvs = UploadsVectorSearch.from_folder(uploads_dir, device=vector_device, max_docs=1000)
        except TypeError:
            # Fallback: attempt without max_docs if the signature differs.
            try:
                uvs = UploadsVectorSearch.from_folder(uploads_dir, device=vector_device)
            except Exception as e2:
                app.logger.warning(f"[/ask] Uploads VS error: {e2}")
                uvs = None
        except Exception as e:
            app.logger.warning(f"[/ask] Uploads VS error: {e}")
            uvs = None

        if uvs is not None:
            hits = uvs.search(question, k=8)
            lines = []
            for i, h in enumerate(hits, start=1):
                txt = _s(h.get("text") or "")
                title = _s(h.get("title") or "")
                sect = _s(h.get("section") or "")
                page = h.get("page")
                path = _s(h.get("path") or "")
                head = f"[U{i}] {title}" if title else f"[U{i}]"
                if sect: head += f" — {sect}"
                if page is not None: head += f" (p.{page})"
                if path: head += f" — {path}"
                if txt:
                    lines.append(head + "\n" + txt.strip()[:1200])
            uploads_ctx = "\n\n".join(lines)
    except Exception as e:
        app.logger.warning(f"[/ask] uploads_ctx warn: {e}")

    # ----------------- DuckDB (LOOKUP) → table context -----------------
    table_ctx = ""
    table_refs = []
    if LOOKUP is not None:
        try:
            hits_tbl = LOOKUP.query(question, topk=5)
            rows = hits_tbl.to_dict(orient="records")
            lines = []
            for i, row in enumerate(rows, start=1):
                solvent = row.get("solvent") or row.get("solvent_system")
                temp = row.get("temp_C") or row.get("temperature_C")
                time_h = row.get("time_h") or row.get("duration_h")
                note = row.get("notes") or ""
                line = f"[T{i}] solvent={_s(solvent)}; temp_C={_s(temp)}; time_h={_s(time_h)}; {_s(note)}".strip()
                lines.append(line)
                url = row.get("url") or (row.get("doi") and f"https://doi.org/{row['doi']}")
                if url:
                    table_refs.append({"title": f"Table row {i}", "url": url})
            table_ctx = "\n".join(lines)
        except Exception as e:
            app.logger.warning(f"[/ask] LOOKUP query error: {e}")

    def _split_reasoning(raw: str) -> tuple[str, str]:
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

    def _ref_url(r: dict) -> str:
        if r.get("url"):
            return r["url"]
        if r.get("doi"):
            return f"https://doi.org/{r['doi']}"
        return ""


    # ---- Reference normalization + matching helpers ----
    _DOI_RX = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.I)

    def _norm_doi(x):
        if not x: 
            return ""
        if isinstance(x, (list, tuple)):
            x = " ".join(map(str, x))
        s = str(x)
        m = _DOI_RX.search(s)
        return m.group(1).lower() if m else ""

    def _norm_url(url, doi=""):
        u = (url or "").strip()
        if u:
            return u
        d = (doi or "").strip().lower()
        return (f"https://doi.org/{d}" if d else "")

    def _canon_title(t):
        if not t:
            return ""
        s = re.sub(r'\s+', ' ', str(t)).strip().lower()
        s = re.sub(r'[^a-z0-9\s]+', '', s)
        return s

    def _safe_year(val):
        if not val:
            return ""
        s = str(val)
        if len(s) >= 4 and s[:4].isdigit():
            return s[:4]
        m = re.search(r'(\d{4})', s)
        return m.group(1) if m else ""

    def _best_ref_idx_for_hit(hit, ref_index, ref_titles, fuzzy_thr=0.84):
        meta = hit.get("meta", {}) if isinstance(hit, dict) else {}
        text = hit.get("text") or ""
        # DOI
        doi_raw = meta.get("doi") or meta.get("paper_id") or meta.get("url") or ""
        doi = _norm_doi(doi_raw) or _norm_doi(text)
        if doi:
            key = ("doi", doi)
            if key in ref_index:
                return ref_index[key]
        # URL
        url = (meta.get("url") or meta.get("oa_url") or meta.get("pdf_url") or "").strip().lower()
        if url:
            key = ("url", url)
            if key in ref_index:
                return ref_index[key]
        # Title
        title = _canon_title(meta.get("title") or "")
        if title:
            key = ("title", title)
            if key in ref_index:
                return ref_index[key]
            # Fuzzy title
            import difflib as _difflib
            best_idx, best = None, 0.0
            for idx, rt in ref_titles.items():
                if not rt:
                    continue
                score = _difflib.SequenceMatcher(a=title, b=rt).ratio()
                if score > best:
                    best, best_idx = score, idx
            if best_idx is not None and best >= fuzzy_thr:
                return best_idx
        return None

    # --------------- on-demand harvest helpers (inner) ---------------
    def _needs_more(hits: list[dict]) -> bool:
        if not hits:
            return True
        # prefer meta identifiers (doi/url/title)
        def _hk(h):
            m = h.get("meta", {}) if isinstance(h, dict) else {}
            doi = (m.get("doi") or m.get("paper_id") or "").strip().lower()
            doi = doi if doi.startswith("10.") else ""
            url = (m.get("url") or m.get("oa_url") or m.get("pdf_url") or "").strip().lower()
            title = (m.get("title") or "").strip().lower()
            return doi or url or title
        uniq = {_hk(h) for h in hits if _hk(h)}
        if len(uniq) < 1:
            return True
        scores = [float(h.get("score", 0.0)) for h in hits[:3]]
        if scores and sum(scores) / len(scores) < 0.18:
            return True
        total_ctx = sum(len(_s(h.get("text",""))) for h in hits)
        return total_ctx < 800
    def _expand_queries(q: str) -> list[str]:
        seeds = [
            "hydrothermal","solvothermal","sol-gel","calcination","anneal",
            "spin-coating","precursor","coprecipitation","microwave",
            "template","electrospinning","nanoparticle","thin film","oxide"
        ]
        base = q.strip()
        out = [base] + [f"{base} {w}" for w in seeds]
        seen, uniq = set(), []
        for s in out:
            if s not in seen:
                seen.add(s); uniq.append(s)
        return uniq[:6]

    def _harvest_reindex(queries: list[str], use_grobid: bool | None = None) -> list[dict]:
        """
        Harvest new papers for the given queries and rebuild the retriever index.
        Returns a list of reference dicts extracted from the chosen bundle so the agent
        can cite them immediately (even before/independent of retriever hits).
        """
        import tempfile, os, sys, subprocess, json

        ROOT = Path(__file__).resolve().parent
        out_dir = ROOT / "harvester" / "out_auto"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ------------- config -------------
        raw = os.getenv("HARVEST_MAX_RESULTS", "6")
        try:
            max_results = int(raw)
        except ValueError:
            max_results = 6
        cfg = (
            "out_dir: {od}\n"
            "queries:\n{qs}\n"
            "since_year: 2016\n"
            f"max_results_per_source: {max_results}\n"
            "grobid_url: http://127.0.0.1:8070\n"
            "unpaywall_email: \"\"\n"
        ).format(
            od=str(out_dir).replace("\\", "/"),
            qs="\n".join(f"- {json.dumps(q)}" for q in queries),
        )

        env = os.environ.copy()
        if use_grobid is None:
            use_grobid = env.get("USE_GROBID", "0").lower() in {"1", "true", "yes"}
        env["USE_GROBID"] = "1" if use_grobid else "0"
        for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            env.setdefault(var, "1")

        def _stream(cmd: list[str]) -> int:
            app.logger.info(f"[harvest_reindex] running: {' '.join(cmd)}")
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=str(ROOT)
            )
            assert p.stdout is not None
            for line in p.stdout:
                sys.stdout.write(line)
            rc = p.wait()
            if rc == 0:
                app.logger.info(f"[harvest_reindex] {' '.join(cmd)} OK")
            else:
                app.logger.warning(f"[harvest_reindex] {' '.join(cmd)} EXIT {rc}")
            return rc

        def _file_has_lines(path: Path, min_lines: int = 1) -> bool:
            try:
                with path.open("r", encoding="utf-8") as f:
                    for i, _ in enumerate(f, 1):
                        if i >= min_lines:
                            return True
                return False
            except FileNotFoundError:
                return False

        # --------- 1) harvest ---------
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            tf.write(cfg)
            cfg_path = tf.name

        rc = _stream(["python", str(ROOT/"harvester/harvester.py"), "--config", cfg_path])
        bundle_raw = out_dir / "bundle.jsonl"
        partial_ok = _file_has_lines(bundle_raw, 1)

        if rc != 0 and not partial_ok:
            app.logger.warning("[harvest_reindex] harvest failed and no bundle produced; skipping index.")
            return []

        if rc != 0 and partial_ok:
            app.logger.info("[harvest_reindex] harvest non-zero, but bundle exists — continuing with index.")

        # --------- 2) add fallback (methods) ---------
        merged_bundle = out_dir / "bundle_with_methods.jsonl"
        _stream([
            "python", str(ROOT/"scripts/bundle_add_fallback.py"),
            str(bundle_raw), str(merged_bundle)
        ])

        # --------- 3) choose bundle + text_key ---------
        # (Assumes BUNDLE_AUTO/MERGED/PLAIN/INDEX_DIR defined at module top)
        bundle_for_index = None
        text_key = "methods"
        if BUNDLE_AUTO.exists():
            bundle_for_index = BUNDLE_AUTO
        elif (ROOT / "out" / "bundle_with_methods.jsonl").exists():
            bundle_for_index = ROOT / "out" / "bundle_with_methods.jsonl"
        else:
            # fall back to the ones wrote in out_auto
            if merged_bundle.exists():
                bundle_for_index = merged_bundle
            elif bundle_raw.exists():
                bundle_for_index = bundle_raw

        if bundle_for_index is None:
            app.logger.warning("[harvest_reindex] No bundle found to index.")
            return []

        # --------- 4) index ---------
        doc_dir = os.getenv("RETRIEVER_INDEX_DIR_DOC")
        pas_dir = os.getenv("RETRIEVER_INDEX_DIR_PASSAGE")

        if doc_dir or pas_dir:
            # doc-level index 
            if doc_dir:
                _stream([
                    "python", str(ROOT/"retriever/index_jsonl.py"),
                    "--bundle", str(bundle_for_index),
                    "--index_dir", str(doc_dir),
                    "--text-key", "abstract"
                ])
            # passage-level index (methods)
            if pas_dir:
                _stream([
                    "python", str(ROOT/"retriever/index_jsonl.py"),
                    "--bundle", str(bundle_for_index),
                    "--index_dir", str(pas_dir),
                    "--text-key", "methods"
                ])
            app.logger.info(f"[harvest_reindex] indexed dual: doc={doc_dir} passage={pas_dir}")
        else:
            # single-index fallback
            _stream([
                "python", str(ROOT/"retriever/index_jsonl.py"),
                "--bundle", str(bundle_for_index),
                "--index_dir", str(INDEX_DIR),
                "--text-key", "methods",
            ])
            app.logger.info(f"[harvest_reindex] indexed single → {INDEX_DIR}")

        # --------- 5) ping retriever ---------
        try:
            with httpx.Client(timeout=20) as s:
                s.post(f"{RETRIEVER_URL.rstrip('/')}/reload")
        except Exception:
            pass

        # --------- 6) BUILD REFERENCE TABLE from the chosen bundle ---------
        def _mk_ref(rec: dict) -> dict:
            def _author_names(auths):
                out = []
                if isinstance(auths, list):
                    for a in auths:
                        if isinstance(a, str):
                            out.append(a)
                        elif isinstance(a, dict):
                            n = (
                                a.get("name")
                                or " ".join(x for x in [a.get("first"), a.get("last")] if x)
                                or " ".join(x for x in [a.get("given"), a.get("family")] if x)
                            )
                            if n:
                                out.append(n)
                return out

            title = (rec.get("title") or rec.get("name") or "").strip()
            paper_id = str(rec.get("paper_id") or "")
            doi = (rec.get("doi") or (paper_id if paper_id.startswith("10.") else "") or "").strip()
            url = (
                rec.get("url") or rec.get("oa_url") or rec.get("pdf_url")
                or (f"https://doi.org/{doi}" if doi else "")
                or ""
            ).strip()
            # year: try explicit, else parse YYYY out of date-like fields
            year = rec.get("year") or rec.get("publication_year")
            if not year:
                for k in ("date", "published", "pub_date"):
                    v = rec.get(k)
                    if isinstance(v, str) and len(v) >= 4 and v[:4].isdigit():
                        year = v[:4]
                        break
            year = str(year or "")
            authors = rec.get("authors") or rec.get("authorships") or rec.get("metadata", {}).get("authors") or []
            authors = _author_names(authors) or authors  # normalize to list of strings if possible

            return {
                "title": title,
                "year": year,
                "url": url,
                "doi": doi,
                "authors": authors,
                "biblio": {},
            }

        harvest_refs: list[dict] = []
        try:
            with bundle_for_index.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 40:  # cap to keep prompt small
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ref = _mk_ref(rec)
                    if ref.get("title"):
                        harvest_refs.append(ref)
        except Exception as e:
            app.logger.warning(f"[/harvest_reindex] refs build warn: {e}")

        return harvest_refs

    # ----------------- KB search + fetch -----------------
    kb_ctx = ""
    kb_refs_raw = []
    kb_hits = []

    try:
        if _h_kb_search:
            # helpers.kb_search returns List[Hit] (text, score, meta)
            kb_hits = _h_kb_search(question, top_k=kb_k) or []
        else:
            kb_hits = []
    except Exception as e:
        app.logger.warning(f"[/ask] KB search failed: {e}")
        kb_hits = []

    def _mk_kb_ref_from_hit(h) -> dict:
        meta = getattr(h, "meta", {}) if not isinstance(h, dict) else h
        j = meta.get("json") if isinstance(meta, dict) else None
        title = ""
        if isinstance(j, dict):
            title = j.get("title") or j.get("name") or ""
        return {
            "title": _s(title) or _s(meta.get("title") or "(KB item)"),
            "year": _s(meta.get("year") or ""),
            "url": _s(meta.get("url") or ""),
            "doi": _s(meta.get("doi") or ""),
            "authors": meta.get("authors") or [],
            "biblio": {},
        }

    kb_refs_raw = [_mk_kb_ref_from_hit(h) for h in kb_hits]
    if kb_refs_raw:
        kb_lines = [f"[KB{i}] {r['title']}" for i, r in enumerate(kb_refs_raw, 1)]
        kb_ctx = "\n".join(kb_lines)

    # ----------------- Web/hybrid retriever (initial) -----------------
    try:
        hits = retriever_search(
            question, k=web_k, level=retriever_level,
            k_doc=k_doc, k_passage=k_passage, w_doc=w_doc, w_passage=w_passage
        ) or []
    except Exception as e:
        app.logger.warning(f"[/ask] retriever_search error: {e}")
        hits = []
    # If evidence thin, optionally auto-harvest -> reindex -> reload -> retry once
    harvest_refs = []  
    if allow_fetch and _needs_more(hits):
        try:
            harvest_refs = _harvest_reindex(_expand_queries(question)) or []   # <— collect refs
            hits = retriever_search(question, k=web_k, level=retriever_level, k_doc=k_doc, k_passage=k_passage, w_doc=w_doc, w_passage=w_passage) or []
        except Exception as e:
            app.logger.warning(f"[/ask] auto-harvest failed: {e}")


    # A) Build web_refs from hit.meta with aggressive normalization (extract DOI from url/text if needed)
    def _hit_meta(h): return h.get("meta", {}) if isinstance(h, dict) else {}
    web_refs = []
    for h in hits:
        m = _hit_meta(h)
        doi = _norm_doi(m.get("doi") or m.get("paper_id") or m.get("url") or h.get("text") or "")
        url = _norm_url(m.get("url") or m.get("oa_url") or m.get("pdf_url"), doi)
        title = _s(m.get("title") or "(no title)")
        year = _safe_year(m.get("year") or m.get("publication_year") or m.get("date") or "")
        web_refs.append({
            "title": title,
            "year":  year,
            "url":   url,
            "doi":   doi,
            "authors": m.get("authors") or [],
            "biblio": {},
        })

    # B) Collect raw references from web, KB, and harvest, then deduplicate and rerank them using ref_utils.
    raw_refs = web_refs + kb_refs_raw + harvest_refs

    # Deduplicate + rerank with ref_utils
    try:
        refs_all = dedupe_and_rerank(
            question, raw_refs, domain_terms=DEFAULT_NANOCHEM_TERMS, top_k=max(40, len(raw_refs))
        )
        if not refs_all:
            refs_all = list(raw_refs)
    except Exception:
        refs_all = list(raw_refs)

    # Numbered REFERENCES string shown to the LLM
    refs_prompt = "\n".join(
        f"[{i+1}] {(r.get('title') or '(no title)')} ({r.get('year') or ''}) — {_ref_url(r)}"
        for i, r in enumerate(refs_all)
    ).strip() or "(no references found)"

    
    # Index references for aligning web snippets (prefer DOI, then URL, then title)
    def _normkey_ref(r: dict):
        doi = _norm_doi(r.get("doi") or r.get("url") or r.get("title"))
        if doi:
            return ("doi", doi)
        url = (r.get("url") or "").strip().lower()
        if url:
            return ("url", url)
        title = _canon_title(r.get("title"))
        if title:
            return ("title", title)
        return None

    ref_index = {}
    ref_titles = {}
    for i, r in enumerate(refs_all, start=1):
        ref_titles[i] = _canon_title(r.get("title"))
        key = _normkey_ref(r)
        if key:
            ref_index[key] = i

    web_ctx_lines = []
    unmatched_debug = []
    for h in hits[:5]:
        idx = _best_ref_idx_for_hit(h, ref_index, ref_titles, fuzzy_thr=float(os.getenv("FUZZY_TITLE_THRESHOLD", "0.84")))
        head = f"[{idx}]" if idx else "[?]"
        title = _s((h.get("meta") or {}).get("title") or "(no title)")
        snip  = _s(h.get("text") or "")
        if snip:
            web_ctx_lines.append(f"{head} {title}\n{snip[:1000]}")
        if not idx:
            try:
                unmatched_debug.append({
                    "meta_title": (h.get("meta") or {}).get("title"),
                    "meta_doi": (h.get("meta") or {}).get("doi") or (h.get("meta") or {}).get("paper_id"),
                    "meta_url": (h.get("meta") or {}).get("url") or (h.get("meta") or {}).get("oa_url") or (h.get("meta") or {}).get("pdf_url")
                })
            except Exception:
                pass
    if unmatched_debug:
        app.logger.info(f"[ask] unmatched hits (could not map to numbered refs): {json.dumps(unmatched_debug)[:800]}")
    web_ctx = "\n\n".join(web_ctx_lines)


    
    # ----------------- attachments → per-question context -----------------
    attachments_ctx = ""
    try:
        atch_ids = []
        payload_json = request.get_json(silent=True) if request.is_json else None
        if isinstance(payload_json, dict):
            atch_ids = payload_json.get("attachments") or []
        if not atch_ids:
            atch_ids = request.form.getlist("attachments") or (
                (request.form.get("attachments") or "").split(",") if request.form.get("attachments") else []
            )
        atch_ids = [(a or "").strip() for a in atch_ids if (a or "").strip()]

        if atch_ids:
            lines = []
            qtext = (payload_json or {}).get("question") or request.values.get("question") or ""
            for j, aid in enumerate(atch_ids, start=1):
                p_txt = ATTACH_DIR / f"{aid}.txt"
                txt = ""
                if p_txt.exists():
                    try:
                        txt = p_txt.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        txt = ""
                else:
                    for pth in ATTACH_DIR.glob(f"{aid}__*"):
                        if pth.suffix.lower() == ".pdf":
                            txt, _ = _extract_pdf_text(pth, max_pages=int(os.environ.get("ATTACH_MAX_PAGES", "40") or "40"))
                        else:
                            try:
                                txt = pth.read_text(encoding="utf-8", errors="ignore")
                            except Exception:
                                txt = ""
                        break
                if txt:
                    for k, ch in enumerate(_best_chunks_from_text(txt, qtext, top_k=3), start=1):
                        head = f"[A{j}.{k}] attachment:{aid}"
                        lines.append(head + "\n" + ch.strip()[:1200])
            attachments_ctx = "\n\n".join(lines)
    except Exception as e:
        try: app.logger.warning(f"[/ask] attachments_ctx warn: {e}")
        except Exception: pass

# ----------------- attachments → per-question context -----------------
    attachments_ctx = ""
    try:
        # 1) ids from JSON or form
        atch_ids = []
        payload_json = request.get_json(silent=True) if request.is_json else None
        if isinstance(payload_json, dict):
            atch_ids = payload_json.get("attachments") or []
        if not atch_ids:
            atch_ids = request.form.getlist("attachments") or ((request.form.get("attachments") or "").split(",") if request.form.get("attachments") else [])
        atch_ids = [(a or "").strip() for a in atch_ids if (a or "").strip()]

        # 2) fallback: use the most recent attachment if none were passed
        if not atch_ids:
            latest_txt = None
            try:
                latest_txt = max(ATTACH_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, default=None)
            except Exception:
                latest_txt = None
            if latest_txt:
                atch_ids = [latest_txt.stem]  # id without .txt

        # 3) build chunks
        if atch_ids:
            lines = []
            qtext = (payload_json or {}).get("question") or request.values.get("question") or ""
            for j, aid in enumerate(atch_ids, start=1):
                p_txt = ATTACH_DIR / f"{aid}.txt"
                txt = ""
                if p_txt.exists():
                    try: txt = p_txt.read_text(encoding="utf-8", errors="ignore")
                    except Exception: txt = ""
                else:
                    for pth in ATTACH_DIR.glob(f"{aid}__*"):
                        if pth.suffix.lower() == ".pdf":
                            txt, _ = _extract_pdf_text(pth, max_pages=int(os.environ.get("ATTACH_MAX_PAGES", "40") or "40"))
                        else:
                            try: txt = pth.read_text(encoding="utf-8", errors="ignore")
                            except Exception: txt = ""
                        break
                if txt:
                    for k, ch in enumerate(_best_chunks_from_text(txt, qtext, top_k=3), start=1):
                        head = f"[A{j}.{k}] attachment:{aid}"
                        lines.append(head + "\n" + ch.strip()[:1200])
            attachments_ctx = "\n\n".join(lines)
        app.logger.info(f"[ask] attachments used: {atch_ids} | ctx_chars={len(attachments_ctx)}")
    except Exception as e:
        try: app.logger.warning(f"[/ask] attachments_ctx warn: {e}")
        except Exception: pass
        
# ----------------- Compose CONTEXT -----------------
    ctx_parts = []
    if attachments_ctx:
        ctx_parts.insert(0, "<<<CTX_ATTACH>>>\n" + attachments_ctx)

    if uploads_ctx: ctx_parts.append("<<<CTX_UPLOADS>>>\n" + uploads_ctx)
    if table_ctx:   ctx_parts.append("<<<CTX_TABLE>>>\n" + table_ctx)
    if kb_ctx:      ctx_parts.append("<<<CTX_KB>>>\n" + kb_ctx)
    if web_ctx:     ctx_parts.insert(0, "<<<CTX_WEB>>>\n" + web_ctx) 
    context_joined = "\n\n---\n\n".join(ctx_parts).strip()

    # ----------------- Prompting -----------------
    # Adjust scale requirements based on attachment presence
    if attachments_ctx:
        # If attachments are present, preserve their scale
        robot_rules = (
            " - Return a discrete lab protocol preserving the scale and quantities from the attached documents.\n"
            " - If the attachment contains specific quantities, maintain those exact amounts.\n"
            " - Include specific masses (mg, g) or mmol for reagents; volumes (mL, L) for liquids as shown in attachments.\n"
            " - Specify temperatures (°C), ramp rates (°C/min), and hold times (min/h) matching the attachment when provided.\n"
            " - Include workup and purification (quench, washing/centrifugation, drying) with volumes from the attachment.\n"
            " - No placeholders (avoid \"e.g.\"/\"or\"). Be decisive.\n"
            " - Avoid using Schlenk line, air-free techniques. Do not suggest inert gas.\n"
            " - Do not output a REFERENCES block in the answer."
        )
    else:
        robot_rules = (
            " - Return a discrete lab protocol with exact quantities on a small scale (e.g., ~0.5–1 mmol of the metal precursor).\n"
            " - Include specific masses (mg) or mmol for reagents; volumes (mL) for liquids.\n"
            " - Specify temperatures (°C), ramp rates (°C/min), and hold times (min/h).\n"
            " - Include workup and purification (quench, washing/centrifugation, drying) with volumes.\n"
            " - No placeholders (avoid \"e.g.\"/\"or\"). Be decisive.\n"
            " - Avoid using Schlenk line, air-free techniques. Do not suggest inert gas.\n"
            " - Do not output a REFERENCES block in the answer."
        )
    reasoning_rules = (
        " - Provide a mechanistic explanation and design considerations for the target.\n"
        " - Focus on: nucleation vs growth; ligand/solvent coordination; " 
        " - IMPORTANT: specify why certain precursors over others; and IMPORTANT: why certain reagents over others.\n"
        " - Do NOT say generic statements, or say you only chose things because they were in context or references. Specify your reasoning.\n"
        " - Do NOT return a step-by-step protocol. Be concise but specific."
    )
    inline_rule = (
        " - When you pull a fact from any numbered REFERENCE, put its number in square brackets right after the sentence "
        " - (e.g. “hydrothermal at 200 °C [3]”)."
    )
    acs_rule = (
        " - Use inline numeric citations ([n]) for any facts taken from REFERENCES.\n"
        " - Do NOT output a REFERENCES block; it will be assembled server-side."
    )

    def strip_references_block(text: str) -> str:
        return re.sub(r"##\s*References[\s\S]*", "", text, flags=re.I).strip()

    if mode == "reasoning":
        prompt = (
            "You are NanoChemGPT. Use the CONTEXT and numbered REFERENCES.\n"
            "Rules:\n"
            " - Prefer CONTEXT and REFERENCES over general knowledge when relevant.\n"
            " - For each bullet, quote or paraphrase a specific finding from CONTEXT or REFERENCES, and cite the source. Do not generalize or invent citations.\n"
            " - Be very specific to the question.\n"
            f"{inline_rule}\n"
            " - If CONTEXT is insufficient, say so explicitly before generalizing.\n"
            " - Every claim that uses info from CONTEXT/REFERENCES must end with [n]. If no [n] applies, omit the claim.\n"
            f"{reasoning_rules}\n"
            f"{acs_rule}\n"
            "Return exactly ONE block:\n"
            "## Mechanistic reasoning\n"
            "- bullet points with inline [n] where appropriate.\n\n"
            f"CONTEXT:\n{context_joined}\n\n"
            f"REFERENCES:\n{refs_prompt}\n\n"
            f"User question: {question}"
        )
    else:
        prompt = (
            "You are NanoChemGPT. Use the CONTEXT and the numbered REFERENCES to propose a synthesis.\n"
            "Rules:\n"
            " - Prefer CONTEXT and REFERENCES over general knowledge when relevant.\n"
            " - For each step, quote or paraphrase a specific finding from CONTEXT or REFERENCES, and cite the source. Do not generalize or invent citations.\n"
            f"{inline_rule}\n"
            " - If CONTEXT is insufficient, say so explicitly before generalizing.\n"
            " - Every claim that uses info from CONTEXT/REFERENCES must end with [n]. If no [n] applies, omit the claim.\n"
            f"{robot_rules}\n"
            f"{acs_rule}\n"
            "Return two blocks exactly in this order:\n"
            "## Synthesis Protocol:\n"
            "1. **Hardware & Glassware**:\n[]\n"
            "2. **Materials**:\n[]\n"
            "3. **Procedure**\n[]\n\n"
            "```reason\n"
            "Every claim that uses info from CONTEXT/REFERENCES must end with [n]. If no [n] applies, omit the claim.\n"
            "Keep rationales terse, but specific to the question, citing references and explaining logic.\n"
            "Add NO other blocks of text.\n"
            "```\n\n"
            f"CONTEXT:\n{context_joined}\n\n"
            f"REFERENCES:\n{refs_prompt}\n\n"
            f"User question: {question}"
        )

    if client is None:
        return jsonify({"ok": False, "error": "OpenAI client not configured"}), 500

    raw = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    ).choices[0].message.content

    # Split answer/rationale and strip any in-answer references block
    if mode == "reasoning":
        answer = strip_references_block(_s(raw))
        rationale = ""
    else:
        a, r = _split_reasoning(strip_references_block(_s(raw)))
        answer, rationale = _s(a), _s(r)

    # ---------- Build references payload (using deduped refs_all) ----------
    
    # ---------- Build references payload (single-pass; no extra dedupe) ----------
    # We already deduped into `refs_all`. Figure out which were cited and prepare used subset.
    def _extract_used_ref_indexes_safe(ans: str, rat: str) -> list[int]:
        try:
            return [int(x) for x in extract_used_ref_indexes(ans, rat) if str(x).isdigit()]
        except Exception:
            return []

    used_idxs = _extract_used_ref_indexes_safe(answer, rationale)
    used_idxs = [i for i in used_idxs if 1 <= i <= len(refs_all)]

    # If no citations but answer exists, ask model to add [n] and retry once
    if not used_idxs and answer:
        fix_prompt = (
            "Add numeric citations [n] to the answer using the numbered REFERENCES. "
            "Do not change wording; only append [n] to sentences that clearly use info from CONTEXT/REFERENCES. "
            "If you cannot justify a sentence by CONTEXT/REFERENCES, leave it without [n].\n\n"
            f"CONTEXT:\n{context_joined}\n\n"
            f"REFERENCES:\n{refs_prompt}\n\n"
            f"ANSWER:\n{answer}"
        )
        try:
            fixed = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.0
            ).choices[0].message.content
            if isinstance(fixed, str) and fixed.strip():
                answer = fixed
                used_idxs = _extract_used_ref_indexes_safe(answer, "")
                used_idxs = [i for i in used_idxs if 1 <= i <= len(refs_all)]
        except Exception as _e:
            pass

    # Split to used refs and create a renumbering map
    try:
        refs_used, index_map = split_used_refs(refs_all, used_idxs)
    except Exception:
        refs_used, index_map = list(refs_all), {i+1: i+1 for i in range(len(refs_all))}

    # Renumber in-text citations to the compact used set
    try:
        answer    = renumber_citations(answer, index_map)
        rationale = renumber_citations(rationale, index_map)
    except Exception:
        pass

    # Build ACS-style block from used refs
    try:
        references_block = format_references_block(refs_used)
    except Exception:
        references_block = ""

    # Usage markers
    try:
        markers = _extract_used_markers(answer, rationale)
    except Exception:
        markers = {"refs": [], "tags": {}, "has_ctx": False}

    # Normalize used indexes to new numbering
    try:
        used_idxs = sorted({ index_map.get(i, i) for i in used_idxs })
    except Exception:
        used_idxs = used_idxs or []

    # Build a lightweight refs_payload for downstream consumers (no double-dedupe)
    refs_payload = {
        "refs_all": refs_all,
        "refs_used": refs_used,
        "index_map": index_map,
    }


    # judge based on helper if present; otherwise simple rule
    if callable(_h_judge_sufficiency):
        try:
            import inspect
            fn = _h_judge_sufficiency
            judged_ok = False
            last_err = None
            # ---- Try signature-aware keyword call (question positional; rest filtered) ----
            try:
                sig = inspect.signature(fn)
                param_names = [p.name for p in sig.parameters.values()]
                accepted = set(param_names) - {'question','q'}
                kw_final = {}
                # thresholds
                if 'min_hits' in accepted: kw_final['min_hits'] = MIN_HITS
                elif 'hits' in accepted: kw_final['hits'] = MIN_HITS
                if 'min_score' in accepted: kw_final['min_score'] = MIN_SCORE
                elif 'score' in accepted: kw_final['score'] = MIN_SCORE
                if 'min_chars' in accepted: kw_final['min_chars'] = MIN_CHARS
                elif 'chars' in accepted: kw_final['chars'] = MIN_CHARS
                # context-like parameter (only if explicitly accepted)
                if 'context' in accepted: kw_final['context'] = context_joined
                elif 'ctx' in accepted: kw_final['ctx'] = context_joined
                elif 'text' in accepted: kw_final['text'] = context_joined
                judged_ok = bool(fn(question, **kw_final))
            except TypeError as te_sig:
                last_err = te_sig
            except Exception as ex_sig:
                last_err = ex_sig
            # ---- If not ok, try <=4 positional variants, no kwargs ----
            if not judged_ok:
                attempts_pos = [
                    [question],
                    [question, context_joined],
                    [question, MIN_HITS],
                    [question, MIN_HITS, MIN_SCORE],
                    [question, MIN_HITS, MIN_SCORE, MIN_CHARS],
                    [question, context_joined, MIN_HITS],
                    [question, context_joined, MIN_HITS, MIN_SCORE],
                ]
                for a in attempts_pos:
                    try:
                        judged_ok = bool(fn(*a))
                        if judged_ok:
                            break
                    except TypeError as te2:
                        last_err = te2
                        continue
                    except Exception as ex2:
                        last_err = ex2
                        continue
            # ---- As a last resort, try safe keyword-only variants WITHOUT any context key first ----
            if not judged_ok:
                kw_candidates = [
                    {'min_hits': MIN_HITS, 'min_score': MIN_SCORE, 'min_chars': MIN_CHARS},
                    {'hits': MIN_HITS, 'score': MIN_SCORE, 'chars': MIN_CHARS},
                    # Then context-like keys, attempted only after non-context variants
                    {'min_hits': MIN_HITS, 'min_score': MIN_SCORE, 'min_chars': MIN_CHARS, 'ctx': context_joined},
                    {'min_hits': MIN_HITS, 'min_score': MIN_SCORE, 'min_chars': MIN_CHARS, 'text': context_joined},
                    {'hits': MIN_HITS, 'score': MIN_SCORE, 'chars': MIN_CHARS, 'ctx': context_joined},
                    {'hits': MIN_HITS, 'score': MIN_SCORE, 'chars': MIN_CHARS, 'text': context_joined},
                ]
                for kw in kw_candidates:
                    try:
                        judged_ok = bool(fn(question, **kw))
                        if judged_ok:
                            break
                    except TypeError as te3:
                        last_err = te3
                        continue
                    except Exception as ex3:
                        last_err = ex3
                        continue
            if not judged_ok and last_err is not None:
                app.logger.warning(f"[/ask] judge/enqueue error (helper): {last_err}")
        except Exception as e:
            app.logger.warning(f"[/ask] judge/enqueue error (helper): {e}")
            judged_ok = len(context_joined) >= MIN_CHARS
    else:
        judged_ok = len(context_joined) >= MIN_CHARS
    # judge_hits fallback if not imported
    _judge_hits = None
    try:
        _judge_hits = judge_hits
    except Exception:
        pass
    if not callable(_judge_hits):
        def _judge_hits(hits, min_hits=1, min_score=0.15, min_chars=64):
            return bool(hits) and len(hits) >= min_hits

    try:
        if kb_hits:
            kb_ok = _judge_hits(kb_hits, min_hits=MIN_HITS, min_score=MIN_SCORE, min_chars=MIN_CHARS)
        else:
            kb_ok = False
    except Exception as e:
        app.logger.warning(f"[/ask] judge kb_hits error: {e}")
        kb_ok = False

    web_thin = _needs_more(hits)

    # final sufficiency: any strong signal should allow skipping harvest/enqueue
    sufficient = judged_ok or kb_ok
    if not sufficient or web_thin:
        try:
            enqueue_text_mining_job(question)
            enqueued = True
        except Exception as e:
            app.logger.warning(f"[/ask] enqueue_text_mining_job failed: {e}")

    # ---- Save best-effort to DB ----
    index_map_s         = {str(k): v for k, v in (index_map or {}).items()}
    refs_payload_s      = _stringify_keys(refs_payload or {})
    refs_used_s         = _stringify_keys(refs_used or [])
    refs_all_s          = _stringify_keys(refs_all or [])
    candidates_s        = _stringify_keys(refs_all or [])

    try:
        db = get_db()
        db.qa.insert_one({
            "question": question,
            "intent": intent,
            "mode": mode,
            "created_at": datetime.utcnow(),
            "answer": answer,
            "rationale": rationale,
            "markers": markers,
            "used_ref_indexes": used_idxs,
            "references_block": references_block,
            "refs": refs_all_s,
            "refs_all": refs_all_s,
            "refs_used": refs_used_s,
            "index_map": index_map_s,
            "kb_refs_count": len(kb_refs_raw),
            "web_refs_count": len(web_refs),
            "table_refs": table_refs,
            "context_present": bool(context_joined),
            "mining_enqueued": enqueued,
        })
    except Exception as e:
        app.logger.warning(f"[/ask] DB save warn: {e}")

    response_payload = {
        "ok": True,
        "question": question,
        "intent": intent,
        "mode": mode,
        "answer": answer,
        "rationale": rationale,
        "markers": markers,
        "references_block": references_block,
        # Updated list of citation indices using the renumbered scheme
        "used_ref_indexes": used_idxs,
        # Provide both the full deduped list and the subset of references actually cited. The
        # frontend uses these to populate candidate dropdowns and to render the references.
        "refs": refs_all_s,
        "refs_all": refs_all_s,
        "refs_used": refs_used_s,
        "candidates": candidates_s,
        "index_map": index_map_s,
        "context_present": bool(context_joined),
        "mining_enqueued": enqueued,
    }
    if isinstance(refs_payload_s, dict):
        response_payload.update(refs_payload_s)
    return jsonify(response_payload)

    # ----------------- tiny helpers -----------------
    
def _extract_pdf_text(path: Path, max_pages: int = 40) -> tuple[str, int]:
    try:
        try:
            from pypdf import PdfReader as _PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader as _PdfReader
            except Exception:
                _PdfReader = None
        if _PdfReader is None:
            raise ImportError("pypdf/PyPDF2 not installed")
        reader = _PdfReader(str(path))
        pages = getattr(reader, 'pages', [])
        n = len(pages) or 0
        out = []
        for i, page in enumerate(pages, 1):
            if i > max_pages: break
            try:
                out.append(page.extract_text() or "")
            except Exception:
                out.append("")
        return ("\n".join(out), n or len(out))
    except Exception as e:
        try: app.logger.warning(f"[_extract_pdf_text] {path.name}: {e}")
        except Exception: pass
        return ("", 0)

def _best_chunks_from_text(text: str, query: str, max_chunk_chars: int = 1200, top_k: int = 3):
    import re as _re
    if not text: return []
    q_tokens = {t for t in _re.findall(r"[A-Za-z0-9]{3,}", (query or "").lower())}
    if not q_tokens: q_tokens = set(_re.findall(r"[A-Za-z0-9]{3,}", text.lower()))
    chunks, buf, size = [], [], 0
    for para in text.splitlines():
        if size + len(para) + 1 > max_chunk_chars and buf:
            chunks.append("\n".join(buf)); buf, size = [], 0
        buf.append(para); size += len(para) + 1
    if buf: chunks.append("\n".join(buf))
    def score(s: str) -> int:
        toks = set(_re.findall(r"[A-Za-z0-9]{3,}", s.lower()))
        return sum(1 for t in toks if t in q_tokens)
    return sorted(chunks, key=score, reverse=True)[:top_k]

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
        app.logger.error(f"parse failed: {e}")
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
        app.logger.error(f"parse_upload failed: {e}")
        return jsonify({"ok": False, "error": f"parse_upload failed: {e}"}), 500

@app.post("/clear_uploads")
def clear_uploads_route():
    try:
        vs.clear_uploads()
    except Exception as e:
        app.logger.warning(f"clear_uploads error: {e}")
    return {"status": "uploads cleared"}

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

@app.post("/attach")
def attach():
    files = request.files.getlist("files") or []
    if not files:
        f = request.files.get("file")
        if f: files = [f]
    if not files:
        abort(400, "No files uploaded.")
    items = []
    for f in files:
        if not f or not getattr(f, "filename", ""): 
            continue
        fname = secure_filename(f.filename or "file")
        aid = uuid.uuid4().hex[:12]
        dest = ATTACH_DIR / f"{aid}__{fname}"
        f.save(dest)
        meta = {"id": aid, "filename": fname, "kind": "pdf" if fname.lower().endswith(".pdf") else "file"}
        if meta["kind"] == "pdf":
            txt, n_pages = _extract_pdf_text(dest, max_pages=int(os.environ.get("ATTACH_MAX_PAGES", "40") or "40"))
            (ATTACH_DIR / f"{aid}.txt").write_text(txt, encoding="utf-8")
            meta.update({"n_pages": n_pages, "n_chars": len(txt)})
        items.append(meta)
    return jsonify({"ok": True, "items": items})

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
        if not f.filename:
            continue
        fname = secure_filename(f.filename)
        dest = BUILTIN_DIR / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
        saved.append(fname)
    return jsonify({"ok": True, "files": saved})

@app.get("/healthz")
def healthz():
    return jsonify(ok=True)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        debug=os.getenv("DEBUG", "0") == "1"
    )