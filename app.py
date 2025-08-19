from __future__ import annotations

import os, io, json, re, glob, traceback, threading, tempfile, textwrap, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Set, List, Dict, Optional
from functools import lru_cache

import httpx
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, request, jsonify, abort, render_template,
    send_file
)
from jinja2 import TemplateNotFound
from openai import OpenAI
from werkzeug.utils import secure_filename

# ──────────────── Local modules ──────────────── #
import vector_store as vs
from converter import validate_step, convert_text_to_robot_ops
from mongo_client import get_db, ping as mongo_ping
from decider.intent import classify_intent
from decider.kb import kb_search, kb_fetch
from decider.judge_sufficiency import judge_sufficiency
from decider.miner_queue import enqueue_text_mining_job
from DuckDB.duck_searcher import get_duck_searcher

# ──────────────── Paths/Config ──────────────── #
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
BUILTIN_DIR = Path(os.getenv("BUILTIN_DIR", BASE_DIR / "builtin")).resolve()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "/mnt/data/uploads")).resolve()
LOOKUP_UPLOAD_DIR = Path(os.getenv("LOOKUP_UPLOAD_DIR", "/mnt/data/datasets")).resolve()
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "/mnt/data/index")).resolve()

for d in (BUILTIN_DIR, UPLOADS_DIR, LOOKUP_UPLOAD_DIR, VECTORSTORE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ──────────────── Flask app ──────────────── #
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.config["JSON_AS_ASCII"] = False  # allow UTF-8

# CSRF setup
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
    return dict(csrf_token=generate_csrf)

@app.before_request
def _log_req():
    print(f"[req] {request.method} {request.path}")

# ──────────────── DuckDB setup ──────────────── #
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
        rows = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
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

# ──────────────── OpenAI client ──────────────── #
load_dotenv()
_no_proxy = httpx.Client(trust_env=False, timeout=120.0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY, http_client=_no_proxy) if OPENAI_API_KEY else None

RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://localhost:8000")

def retriever_search(query: str, k: int = 8, mode: str = "hybrid", alpha: float = 0.7) -> list[dict]:
    try:
        r = _no_proxy.post(f"{RETRIEVER_URL}/search", json={"query": query, "k": k, "mode": mode, "alpha": alpha}, timeout=60)
        r.raise_for_status()
        return r.json().get("hits", [])
    except Exception as e:
        print("[retriever] search failed:", e)
        return []

# ──────────────── Utilities ──────────────── #
def _safe_text(x: Any) -> str:
    try:
        return str(x) if x is not None else ""
    except Exception:
        return ""

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

# ---- Citation extraction and formatting helpers ----
_CIT_RX_BRACKET = re.compile(r"\[(\d{1,4})\]")
_CIT_RX_FULL    = re.compile(r"【(\d{1,4})】")
_CIT_RX_FOOT    = re.compile(r"\[\^(\d{1,4})\]")

def _extract_used_ref_indexes(*texts: str) -> list[int]:
    """Return sorted unique numeric citation indexes found like [1], [2], 【3】, or [^4]."""
    seen = set()
    for t in texts:
        if not t:
            continue
        for rx in (_CIT_RX_BRACKET, _CIT_RX_FULL, _CIT_RX_FOOT):
            for m in rx.finditer(t):
                try:
                    seen.add(int(m.group(1)))
                except Exception:
                    pass
    return sorted(seen)

def _format_acs_reference(ref: dict) -> str:
    """Lightweight ACS-ish formatting from a heterogeneous ref dict."""
    def _s(x): return _safe_text(x)
    # Authors
    authors = ref.get("authors") or ref.get("authorships") or []
    names = []
    for a in authors:
        if isinstance(a, dict):
            nm = a.get("name") or a.get("author", {}).get("display_name") or a.get("display_name") or a.get("last_name")
            if nm: names.append(_s(nm))
        elif isinstance(a, str):
            names.append(_s(a))
    if len(names) > 6:
        names = names[:6] + ["et al."]
    authors_str = "; ".join([n for n in names if n])

    # Title
    title = _s(ref.get("title") or ref.get("display_name") or ref.get("paper_title") or "(no title)")

    # Journal / Venue
    journal = ref.get("journal") or ref.get("venue") or ref.get("host_venue") or ref.get("container_title") or ""
    if isinstance(journal, dict):
        journal = journal.get("display_name") or journal.get("name") or journal.get("title") or ""
    journal = _s(journal)

    # Year (derive from date if missing)
    year = ref.get("year") or ref.get("published_year") or ref.get("publication_year")
    if not year:
        pubdate = _s(ref.get("publication_date") or ref.get("published_date"))
        if len(pubdate) >= 4 and pubdate[:4].isdigit():
            year = pubdate[:4]
    year = _s(year)

    # Volume / Issue / Pages
    biblio = ref.get("biblio") or {}
    volume = _s(ref.get("volume") or biblio.get("volume"))
    issue  = _s(ref.get("issue")  or biblio.get("issue"))
    fp     = _s(ref.get("first_page") or biblio.get("first_page") or ref.get("page_start"))
    lp     = _s(ref.get("last_page")  or biblio.get("last_page")  or ref.get("page_end"))
    pages  = ""
    if fp and lp:
        pages = f"{fp}-{lp}"
    elif fp:
        pages = fp

    # DOI/URL
    doi = _s(ref.get("doi"))
    url = _s(ref.get("url"))
    tail = ""
    if doi:
        tail = f"DOI: {doi}"
    elif url:
        tail = url

    parts = []
    if authors_str: parts.append(authors_str + ".")
    if title:       parts.append(title + ".")
    trailer = []
    if journal:     trailer.append(journal)
    if year:        trailer.append(year)
    if volume:      trailer.append(volume if not issue else f"{volume}({issue})")
    if pages:       trailer.append(pages)
    if trailer:
        parts.append(", ".join(trailer) + ".")
    if tail:
        parts.append(tail)
    return " ".join(p for p in parts if p).strip()

def _format_references_block_from_used(used_indexes: list[int], refs: list[dict]) -> str:
    """Create a numbered reference block (1-based) for only the cited refs present in `refs`."""
    if not used_indexes or not refs:
        return ""
    lines = []
    for idx in sorted(set(used_indexes)):
        if not isinstance(idx, int) or idx < 1 or idx > len(refs):
            continue
        lines.append(f"{idx}. " + _format_acs_reference(refs[idx-1]))
    return "\n".join(lines)

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
        mongo_ping()
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
    fname = secure_filename(f.filename)
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
            print("[/upload] get_db failed (continuing without DB):", e)
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

# ---- Search API ---- #
# @app.post("/search")
# def search_route():
#     payload = request.get_json(silent=True) or {}
#     q = (payload.get("q") or payload.get("query") or "").strip()
#     n = int(payload.get("n") or 6)
#     if not q:
#         abort(400, "Missing 'q' (query).")
#     refs = basic_search(q, n)
#     return jsonify({"results": refs})

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
    # ----------------- tiny helpers -----------------
    def _s(x):
        return _safe_text(x)

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
        if r.get("url"): return r["url"]
        if r.get("doi"): return f"https://doi.org/{r['doi']}"
        return ""

    # --------------- on-demand harvest helpers (inner) ---------------
    def _needs_more(hits: list[dict]) -> bool:
        if not hits:
            return True
        uniq = {h.get("paper_id") for h in hits if h.get("paper_id")}
        if len(uniq) < 2:
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

    def _harvest_reindex(queries: list[str]) -> None:
        import os, json, subprocess, tempfile, pathlib
        out_dir = "harvester/out_auto"
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        cfg = "out_dir: {od}\nqueries:\n{qs}\nsince_year: 2016\nmax_results_per_source: 15\ngrobid_url: http://127.0.0.1:8070\nunpaywall_email: \"\"\n".format(
            od=out_dir.replace("\\","/"),
            qs="\n".join(f"- {json.dumps(q)}" for q in queries),
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            tf.write(cfg); cfg_path = tf.name
        # 1) harvest -> out_auto/bundle.jsonl
        subprocess.run(["python","harvester/harvester.py","--config",cfg_path], check=True)
        # 2) merge/refresh working bundle
        subprocess.run(["python","scripts/bundle_add_fallback.py", f"{out_dir}/bundle.jsonl", "out/bundle_with_methods.jsonl"], check=True)
        # 3) rebuild index
        subprocess.run(["python","retriever/index_jsonl.py","--bundle","out/bundle_with_methods.jsonl","--index_dir","retriever/index","--embed-backend","none"], check=True)
        # 4) tell retriever to reload (best-effort)
        try:
            import httpx
            with httpx.Client(timeout=20) as s:
                s.post("http://127.0.0.1:8000/reload")
        except Exception:
            pass

    try:
        payload = request.get_json(silent=True) or {}
        user_mode = _s(payload.get("mode") or "").lower().strip()
        question = _s(payload.get("question")).strip()
        want_inline = bool(payload.get("want_inline_citations", True))
        allow_fetch = bool(payload.get("allow_fetch", True))  # <-- new flag (default True)
        if not question:
            abort(400, "No question.")

        # ----------------- classify intent -----------------
        intent = None
        try:
            intent = classify_intent(question)
        except Exception as e:
            print("[/ask] classify_intent failed:", e)
            intent = None
        intent_str = _s(intent.get("label") if isinstance(intent, dict) else intent)

        # Decide mode from user preference > intent > default
        if user_mode in ("robot","reasoning"):
            mode = user_mode
        elif (intent_str or "").lower() in ("mechanism","explain","reason"):
            mode = "reasoning"
        else:
            mode = "robot"

        # Search breadth from intent (fallbacks are conservative)
        kb_k = 6 if "kb" in (intent_str or "").lower() else 4
        web_k = 8

        # ----------------- uploads vector context -----------------
        try:
            uploads_ctx = _s(vs.search(question, k=8) or "")
        except Exception as e:
            print("[/ask] vs.search error:", e)
            uploads_ctx = ""

        # ----------------- DuckDB context -----------------
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
                    if url: table_refs.append({"title": f"Table row {i}", "url": url})
                table_ctx = "\n".join(lines)
            except Exception as e:
                print("[/ask] LOOKUP query error:", e)

        # ----------------- KB search + fetch -----------------
        kb_ctx = ""
        kb_refs_raw = []
        try:
            try:
                kb_hits = kb_search(question, k=kb_k) or []
            except TypeError:
                kb_hits = kb_search(question) or []
            kb_ids = [h.get("id") for h in kb_hits if isinstance(h, dict) and h.get("id")]
            kb_docs = []
            if kb_ids:
                try:
                    kb_docs = kb_fetch(kb_ids) or []
                except Exception as e:
                    print("[/ask] kb_fetch failed:", e)
                    kb_docs = []

            def _mk_kb_ref(d: dict) -> dict:
                return {
                    "title": _s(d.get("title") or d.get("name") or "(KB item)"),
                    "year": _s(d.get("year") or d.get("publication_year") or ""),
                    "url": _s(d.get("url") or d.get("link") or ""),
                    "doi": _s(d.get("doi") or ""),
                    "authors": d.get("authors") or d.get("authorships") or [],
                    "biblio": d.get("biblio") or {},
                }

            items = kb_docs if kb_docs else kb_hits
            kb_refs_raw = [_mk_kb_ref(d) for d in items if isinstance(d, dict)]
            if kb_refs_raw:
                kb_lines = [f"[KB{i}] {r['title']}" for i,r in enumerate(kb_refs_raw, 1)]
                kb_ctx = "\n".join(kb_lines)
        except Exception as e:
            print("[/ask] KB search failed:", e)

        # ----------------- Web/hybrid retriever (initial) -----------------
        hits = retriever_search(question, k=web_k, mode="hybrid", alpha=0.7) or []

        # If evidence thin, optionally auto-harvest -> reindex -> reload -> retry once
        if allow_fetch and _needs_more(hits):
            try:
                _harvest_reindex(_expand_queries(question))
                hits = retriever_search(question, k=web_k, mode="hybrid", alpha=0.7) or []
            except Exception as e:
                print("[/ask] auto-harvest failed:", e)

        # ---- Convert hits to web_refs ----
        web_refs = []
        for h in hits:
            web_refs.append({
                "title": _s(h.get("title") or "(no title)"),
                "year": "",
                "url": _s(h.get("url") or ""),
                "doi": _s(h.get("paper_id") if (h.get("paper_id","").startswith("10.")) else ""),
                "authors": h.get("authors", []),
                "biblio": {},
            })

        # ----------------- Deduplicate & enumerate references -----------------
        def _normkey(r: dict) -> str:
            return (r.get("doi") or r.get("title") or "").strip().lower()

        refs = []
        seen = set()
        for r in web_refs + kb_refs_raw:
            key = _normkey(r)
            if key and key not in seen:
                refs.append(r)
                seen.add(key)

        refs_prompt = "\n".join(
            f"[{i+1}] {(r.get('title') or '(no title)')} ({r.get('year') or ''}) — {_ref_url(r)}"
            for i, r in enumerate(refs)
        ).strip()

        # ----------------- Compose CONTEXT -----------------
        ctx_parts = []
        if uploads_ctx: ctx_parts.append("<<<CTX_UPLOADS>>>\n" + uploads_ctx)
        if table_ctx:   ctx_parts.append("<<<CTX_TABLE>>>\n" + table_ctx)
        if kb_ctx:      ctx_parts.append("<<<CTX_KB>>>\n" + kb_ctx)
        context_joined = "\n\n---\n\n".join(ctx_parts).strip()

        # ----------------- Prompting -----------------
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
            " - Focus on: nucleation vs growth; ligand/solvent coordination; surfactants; reduction/oxidation; temperature profile and morphology control; atmosphere; pitfalls; safety.\n"
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
            return re.sub(r"##\s*References[\s\S]*", "", text, flags=re.I).strip()

        if mode == "reasoning":
            prompt = (
                "You are NanoChemGPT. Use the CONTEXT and numbered REFERENCES.\n"
                "Rules:\n"
                " - Prefer CONTEXT and REFERENCES over general knowledge when relevant.\n"
                " - For each bullet, quote or paraphrase a specific finding from CONTEXT or REFERENCES, and cite the source. Do not generalize or invent citations.\n"
                " - If you use any content from CONTEXT, append [CTX] on that line.\n"
                f"{inline_rule}\n"
                " - If CONTEXT is insufficient, say so explicitly before generalizing.\n"
                f"{reasoning_rules}\n"
                f"{acs_rule}\n"
                "Return exactly ONE block:\n"
                "## Mechanistic reasoning\n"
                "- bullet points with inline [n] and [CTX] where appropriate.\n\n"
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
                "For each key justification, add inline tags: [CTX] uploaded/context hits, [PARSED] parsed protocols, [n] for numbered web REFERENCES, [GEN] if inferred.\n"
                "Keep rationales terse.\n"
                "Add NO other blocks of text.\n"
                "```\n\n"
                f"CONTEXT:\n{context_joined}\n\n"
                f"REFERENCES:\n{refs_prompt}\n\n"
                f"User question: {question}"
            )

        raw = client.chat.completions.create(
            model="gpt-4o-mini",
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

        # ---- Build a REFERENCES block of only used items ----
        used_idxs = _extract_used_ref_indexes(answer, rationale)
        references_block = _format_references_block_from_used(used_idxs, refs)

        # ---- Usage markers (refs + tags) ----
        markers = _extract_used_markers(answer, rationale)

        # ---- Sufficiency check + enqueue mining if thin (kept) ----
        enqueued = False
        try:
            try:
                sufficient = judge_sufficiency(question, context_joined, refs_count=len(refs))
            except TypeError:
                sufficient = judge_sufficiency(question, context_joined)
            if not bool(sufficient):
                try:
                    enqueue_text_mining_job(question)
                    enqueued = True
                except Exception as e:
                    print("[/ask] enqueue_text_mining_job failed:", e)
        except Exception as e:
            print("[/ask] judge/enqueue error:", e)

        # ---- Save best-effort to DB ----
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
                "refs": refs,
                "kb_refs_count": len(kb_refs_raw),
                "web_refs_count": len(web_refs),
                "table_refs": table_refs,
                "context_present": bool(context_joined),
                "mining_enqueued": enqueued,
            })
        except Exception as e:
            print("[/ask] DB save warn:", e)

        return jsonify({
            "ok": True,
            "question": question,
            "intent": intent,
            "mode": mode,
            "answer": answer,
            "rationale": rationale,
            "markers": markers,
            "used_ref_indexes": used_idxs,
            "references_block": references_block,
            "refs": refs,
            "context_present": bool(context_joined),
            "mining_enqueued": enqueued,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"/ask failed: {e}"}), 500

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
        fname = secure_filename(f.filename)
        dest = BUILTIN_DIR / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
        saved.append(fname)
    return jsonify({"ok": True, "files": saved})

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        debug=os.getenv("DEBUG", "0") == "1"
    )