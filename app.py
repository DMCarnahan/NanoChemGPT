from __future__ import annotations

import os, io, json, re, glob, traceback, threading
from datetime import datetime
from pathlib import Path
from typing import Any, Set
from functools import lru_cache

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

# ──────────────── Local modules ──────────────── #
import vector_store as vs
from converter import validate_step, convert_text_to_robot_ops
from mongo_client import get_db, ping as mongo_ping
from internet_search import search_papers, set_user_agent

# ---------------------------------------------------------------------------
#  Search helpers and reference filtering utilities
#
# These helpers encapsulate logic for performing OpenAlex/Crossref searches,
# applying material/topic-aware filtering and ranking, and formatting
# references. They exist here instead of in internet_search.py to avoid
# circular imports and to keep this file self-contained. The functions
# derive_query_profile and filter_and_rerank_generic implement a simple
# token-based approach to extract chemical formulas and morphology terms
# from the user question. The shim _call_search_papers() handles legacy
# versus new keyword signatures for search_papers().

# A lightweight regex for extracting chemical formulas like Mn3O4, NiO, Fe3O4.
_FORMULA_REGEX = re.compile(r"\b(?:[A-Z][a-z]?\d*){1,5}\b")

# A mapping of known formula aliases to additional synonyms. This is not
# exhaustive but covers a few common oxides encountered in the app.
_MATERIAL_SYNONYMS = {
    "mn3o4": [
        "hausmannite",
        "manganese tetroxide",
        "manganese(ii,iii) oxide",
        "trimanganese tetraoxide",
    ],
    "nio": [
        "nickel oxide",
        "nickel(ii) oxide",
    ],
    "fe3o4": [
        "magnetite",
        "iron(ii,iii) oxide",
        "black iron oxide",
    ],
}

# A small list of shape descriptors. These can be extended as needed.
_SHAPE_TERMS = [
    "nanorod", "nanorods",
    "nanowire", "nanowires",
    "nanotube", "nanotubes",
    "nanoribbon", "nanoribbons",
    "nanobelt", "nanobelts",
    "nanosheet", "nanosheets",
    "nanoplate", "nanoplates",
    "nanoparticle", "nanoparticles",
    "nanocube", "nanocubes",
]

def _normalize_formula(f: str) -> str:
    """Normalize a chemical formula to lowercase without unicode subscripts."""
    return (f or "").strip().lower()

def derive_query_profile(question: str) -> dict:
    """
    Derive a simple profile from the user question consisting of material
    formulas (with synonyms) and shape descriptors. Chemical formulas are
    extracted via regex and converted to lowercase. Known aliases are
    appended from `_MATERIAL_SYNONYMS`. Shape descriptors are detected via
    substring matching against `_SHAPE_TERMS`.

    Returns a dict with keys: 'materials' and 'shapes' each containing
    sets of lowercase strings.
    """

    materials: set[str] = set()
    shapes: set[str] = set()
    q_lower = (question or "").lower()

    # formulas (lowercase) + synonyms
    for f in _extract_formulas(question or ""):
        materials.add(f)
        if f in _MATERIAL_SYNONYMS:
            for alias in _MATERIAL_SYNONYMS[f]:
                materials.add(alias.lower())

    # shapes
    for term in _SHAPE_TERMS:
        if term in q_lower:
            shapes.add(term)

    return {"materials": materials, "shapes": shapes}

def filter_and_rerank_generic(question: str, refs: list[dict]) -> list[dict]:
    """
    Filter and rerank search results based on material and shape tokens
    derived from the question. Each reference is scored according to how
    many material/shape tokens appear in its title or abstract. Material
    matches count as two points; shape matches count as one point. Results
    with a score of zero are discarded if any material tokens were present
    in the query. If no material tokens are present, all results are
    considered but still sorted by score.

    Args:
        question: the user question (string)
        refs: list of reference dicts (from search_papers)
    Returns:
        a list of up to eight references sorted by score, or the original
        refs if filtering returns an empty list.
    """
    if not refs:
        return []
    prof = derive_query_profile(question)
    mats = prof.get("materials", set())
    shapes = prof.get("shapes", set())
    scored: list[tuple[int, dict]] = []
    for r in refs:
        title = (r.get("title") or "").lower()
        abstract = (r.get("abstract") or "").lower()
        blob = f"{title} {abstract}"
        score = 0
        # count material matches
        for m in mats:
            if m and m in blob:
                score += 2
        # count shape matches
        for sh in shapes:
            if sh and sh in blob:
                score += 1
        scored.append((score, r))
    # If materials present, remove zero-score entries
    if mats:
        scored = [s for s in scored if s[0] > 0]
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [r for _, r in scored][:8]
    # If filtering resulted in no entries, return original refs (up to 8)
    if not filtered:
        return refs[:8]
    return filtered

def _call_search_papers(q: str, n: int = 20, aboutness_flag: bool = True) -> list[dict]:
    """
    Compatibility shim for search_papers() to handle both 'use_aboutness'
    and 'aboutness' keyword names. Attempts to call search_papers with
    use_aboutness set to aboutness_flag. If that fails due to a TypeError
    (unexpected keyword), falls back to 'aboutness' or no keyword.
    Returns an empty list on error.
    """
    try:
        return search_papers(q, n=n, use_aboutness=aboutness_flag) or []
    except TypeError:
        try:
            return search_papers(q, n=n, aboutness=aboutness_flag) or []
        except TypeError:
            try:
                return search_papers(q, n=n) or []
            except Exception as e:
                print("[_call_search_papers] search error:", e)
                return []

def _format_acs_reference(r: dict) -> str:
    """
    Format a reference dictionary as an ACS-style reference string.
    This function expects keys: 'authors', 'title', 'journal', 'year',
    'doi', 'url'. Missing fields are handled gracefully.
    """
    authors = r.get("authors") or []
    if isinstance(authors, list):
        names = []
        for a in authors[:6]:
            if isinstance(a, dict):
                # OpenAlex provides dicts with 'author' key
                n = a.get("author", {}).get("display_name", "")
            else:
                n = str(a)
            n = (n or "").strip()
            if n:
                names.append(n)
        auth_str = "; ".join(names)
        if len(authors) > 6:
            auth_str += "; et al."
    else:
        auth_str = str(authors).strip()
    title = (r.get("title") or "(no title)").strip()
    journal = (r.get("journal") or r.get("venue") or "").strip()
    year = str(r.get("year") or "").strip()
    doi = (r.get("doi") or "").strip()
    url = (r.get("url") or "").strip()
    if doi and not doi.startswith("http"):
        doi_str = f"https://doi.org/{doi}"
    else:
        doi_str = url
    parts = [p for p in [auth_str and f"{auth_str}.", f"{title}.", journal, year] if p]
    return (" ".join(parts) + (" " + doi_str if doi_str else "")).strip()
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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=_no_proxy)

# Set the contact email for OpenAlex API via set_user_agent if available.
try:
    # Use environment variables CONTACT_EMAIL or OPENALEX_CONTACT_EMAIL as the contact email.
    contact_email = os.getenv("CONTACT_EMAIL") or os.getenv("OPENALEX_CONTACT_EMAIL")
    if contact_email:
        set_user_agent(contact_email)
except Exception:
    # If set_user_agent isn't available or fails, continue without raising.
    pass

try:
    contact_email = os.getenv("CONTACT_EMAIL") or os.getenv("OPENALEX_CONTACT_EMAIL")
    if contact_email:
        set_user_agent(contact_email)
except Exception:
    pass

# ──────────────── Utilities ──────────────── #
# --- unicode subscript → ascii digits helper ---
_SUB_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
def from_subscript(s: str) -> str:
    return (s or "").translate(_SUB_MAP)

def _extract_formulas(q: str) -> Set[str]:
    """
    Extract chemical formulas; support lowercase & unicode subscripts (e.g., mn₃o₄).
    Returns lowercase tokens (e.g., 'mn3o4').
    """
    s = from_subscript(q or "")
    hits = set(_FORMULA_REGEX.findall(s)) | set(_FORMULA_REGEX.findall(s.upper()))
    return {h.lower() for h in hits}

def _safe_text(x: Any) -> str:
    try:
        return str(x) if x is not None else ""
    except Exception:
        return ""

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

def basic_search(query: str, n: int = 6) -> list[dict]:
    """Merge local LOOKUP hits + OpenAlex web hits; de-dup by doi/title."""
    if not (query or "").strip():
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
                    })
        except Exception as e:
            print("[basic_search] lookup query failed:", e)

    uq = (query or "").strip()
    use_aboutness = len(uq) <= 80 and uq.count(" ") < 12 and not any(p in uq for p in ",;:!?/\\|")

    web = []
    try:
        web = _call_search_papers(query, n=n, aboutness_flag=use_aboutness) or []
    except Exception as e:
        print("[basic_search] internet_search error:", e)

    seen = {(d.get("doi") or d.get("title", "")).lower() for d in local}
    for w in web:
        key = (w.get("doi") or w.get("title", "")).lower()
        if key not in seen:
            local.append({
                k: w.get(k, "") for k in ("title","year","url","doi","abstract","journal","authors")
            } | {"source":"web"})
            seen.add(key)
    return local[:2 * n]

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

@lru_cache(maxsize=128)
def cached_basic_search(q, n):
    try:
        return basic_search(q, n) or []
    except Exception:
        return []

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
@app.post("/search")
def search_route():
    payload = request.get_json(silent=True) or {}
    q = (payload.get("q") or payload.get("query") or "").strip()
    n = int(payload.get("n") or 6)
    if not q:
        abort(400, "Missing 'q' (query).")
    refs = basic_search(q, n)
    return jsonify({"results": refs})

# ---- Ask ---- #
@app.post("/ask")
def ask():
    """
    Main endpoint for generating synthesis protocols and mechanistic reasoning.
    This route performs the following steps:
      1. Extract the question and mode from the request payload.
      2. Gather context from uploads (vector store) and table lookups (DuckDB).
      3. Perform a web search using OpenAlex/Crossref via search_papers. A compatibility
         shim handles both legacy and modern keyword signatures. For long
         natural-language queries, the search skips topic classification
         ('aboutness') to avoid HTTP 400 errors. Results are de-duplicated
         against table hits and then filtered/ranked based on material and
         morphology tokens derived from the query.
      4. If materials are detected and fewer than 3 references remain, an
         enriched secondary search is performed using targeted query rewrites
         (e.g., "<material> nanorod synthesis").
      5. As a final fallback, if no references remain and materials are
         present, a material-only search is executed using only the
         material names plus 'nanorod(s)'.
      6. A numbered reference list is built for prompting and display.
      7. An appropriate prompt is constructed depending on the selected mode:
         "robot" for protocol generation or "reasoning" for mechanistic insight.
      8. The OpenAI model is invoked to generate a response, which is
         post-processed to separate rationale sections, fix citations, and
         insert [CTX] markers. A formatted ACS reference block is appended
         unless the mode is strict.
      9. The answer, rationale, and references are logged in the database
         and returned as JSON.
    """
    def split_reasoning(raw: str) -> tuple[str, str]:
        """Extract rationale blocks and return (answer, rationale)"""
        if not raw:
            return "", ""
        text = raw.strip()
        # Remove code-fenced rationale blocks
        fence_rx = re.compile(r"```(?:reason|rationale|reasoning)\s*([\s\S]*?)```", re.I)
        rationale = ""
        fences = list(fence_rx.finditer(text))
        if fences:
            rationale = fences[-1].group(1).strip()
            answer = fence_rx.sub("", text).strip()
            return answer, rationale
        # Remove heading rationale blocks
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

        # Build unified context string
        context_parts = []
        if vs_ctx:
            context_parts.append("<<<CTX_UPLOADS>>>\n" + vs_ctx)
        if table_ctx:
            context_parts.append("<<<CTX_TABLE>>>\n" + table_ctx)
        context_joined = "\n\n---\n\n".join(context_parts).strip()

        print("[/ask] search impl:", getattr(search_papers, "__module__", None), getattr(search_papers, "__name__", None))

        # Initial web references (skip aboutness for long queries)
        refs = []
        try:
            refs = _call_search_papers(q, n=20, aboutness_flag=False)
        except Exception as e:
            print("[/ask] search_papers error:", e)
        if table_refs:
            refs = list(refs) + table_refs
        # Log initial counts
        print(f"[/ask] initial refs: {len(refs)}")
        _pre_filter_refs = list(refs)

        # Filter and rerank generically
        try:
            refs = filter_and_rerank_generic(q, refs) or []
        except Exception as e:
            print("[/ask] filter error:", e)
        print(f"[/ask] after filter: {len(refs)}")
        # If filter empties results but we had some, restore top raw entries
        if not refs and _pre_filter_refs:
            refs = _pre_filter_refs[:8]
            print(f"[/ask] filter emptied results — restored {len(refs)} raw refs")

        # Enriched re-search if materials exist and refs are sparse
        prof = derive_query_profile(q)
        if prof.get("materials") and len(refs) < 3:
            def _es_local_helper(query: str, materials: set[str], shapes: set[str]) -> list[dict]:
                mats = sorted({m for m in materials if any(ch.isalpha() for ch in m)})[:2]
                shape = next(iter(shapes), "nanorod")
                seeds = [
                    query,
                    f"{' '.join(mats)} {shape} synthesis",
                    f"{' '.join(mats)} hydrothermal {shape}",
                    f"{' '.join(mats)} {shape} preparation",
                ]
                all_refs: list[dict] = []
                seen: set[str] = set()
                for s in seeds:
                    try:
                        hits = _call_search_papers(s, n=12, aboutness_flag=False)
                    except Exception as e:
                        print("[enriched_search] search fail:", e)
                        hits = []
                    for h in hits:
                        key = (h.get("doi") or h.get("title", "")).lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        all_refs.append(h)
                return all_refs[:24]
            try:
                more = _es_local_helper(q, prof["materials"], prof["shapes"])
                if more:
                    refs = filter_and_rerank_generic(q, refs + more) or refs
            except Exception as e:
                print("[/ask] enriched_search error:", e)

        # Final relaxed material-only fallback if still empty
        try:
            prof2 = derive_query_profile(q)
        except Exception:
            prof2 = {}
        if not refs and prof2.get("materials"):
            mats_list = sorted(list(prof2["materials"]))[:2]
            seeds = []
            for mmat in mats_list:
                seeds.extend([f"{mmat} nanorod", f"{mmat} nanorods"])
            more: list[dict] = []
            seen_keys: set[str] = set()
            for s in seeds:
                try:
                    hits = _call_search_papers(s, n=12, aboutness_flag=False) or []
                except Exception as e:
                    print("[/ask] final material-only search error:", e)
                    hits = []
                for h in hits:
                    key = (h.get("doi") or h.get("title", "")).lower()
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    more.append(h)
            if more:
                try:
                    refs = filter_and_rerank_generic(q, more) or []
                except Exception as e:
                    print("[/ask] final filter error:", e)
        # Log final references
        print("[/ask] refs (filtered):")
        for i, r in enumerate(refs or [], 1):
            print(f"  [{i}] {r.get('title')} — {r.get('doi') or r.get('url')}")

        # Build numbered reference prompt
        def _ref_url(r: dict) -> str:
            if r.get("url"):
                return r["url"]
            if r.get("doi"):
                return f"https://doi.org/{r['doi']}"
            return ""
        refs_prompt = "\n".join(
            f"[{i+1}] {(r.get('title') or '(no title)')} ({r.get('year') or ''}) — {_ref_url(r)}"
            for i, r in enumerate(refs)
        ).strip()

        # ----------------- Prompt construction -----------------
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
            answer, rationale = split_reasoning(strip_references_block(raw))

        # Rationale fallback if missing
        if not (rationale or "").strip() and mode != "reasoning":
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

        # Post-pass: enforce citations & CTX usage if missing (non-strict modes)
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

        # Enforce [CTX] usage if missing and context present
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

        # Build a references block and optionally append to answer
        want_refs_block = bool(payload.get("want_reference_block", True))
        is_strict = mode in ("robot_strict", "reasoning_strict")
        refs_block_text = ""
        if want_refs_block and refs:
            refs_block_text = "\n".join(
                f"{i+1}. {_format_acs_reference(r)}" for i, r in enumerate(refs)
            )
        if want_refs_block and not is_strict and refs_block_text:
            answer = f"{(answer or '').rstrip()}\n\n## References\n{refs_block_text}"

        # Persist to DB and return
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
            "references_block": refs_block_text,
            "refs": refs,
            "refs_used": used_summary.get("refs", []),
            "used": used_summary,
            "mode": mode,
            "qa_id": qa_id,
            "ctx_vs": (vs_ctx or "")[:8000],
            "ctx_table": (table_ctx or "")[:4000],
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
