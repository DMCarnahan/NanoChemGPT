import os, io, json, lzma, re, functools, threading, traceback
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, abort, render_template, send_file, make_response
from jinja2 import TemplateNotFound
from openai import OpenAI
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from bson import ObjectId
from flask_wtf.csrf import CSRFProtect, generate_csrf
# ───────────────────────────── Local modules ────────────────────────────── #
import vector_store as vs
from converter import validate_step, validate_file  # line-aware, normalizing validator
from mongo_client import get_db, ping as mongo_ping
from dataset_searcher import load_table, DatasetSearcher
from internet_search import search_papers            # OpenAlex helper
# ─────────────────────────────────────────────────────────────────────────── #

# ─── Directories & global paths ─────────────────────────────────────────── #
BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
BUILTIN_DIR    = Path(os.getenv("BUILTIN_DIR", "/mnt/data/builtin")).resolve()
UPLOADS_DIR    = Path(os.getenv("UPLOADS_DIR", "/mnt/data/uploads")).resolve()
VECTORSTORE_DIR= Path(os.getenv("VECTORSTORE_DIR", "/mnt/data/index")).resolve()
MECH_KB_DIR    = Path(os.getenv("MECH_KB_DIR", BASE_DIR / "mechanistic_kb")).resolve()
MECH_INDEX_DIR = (MECH_KB_DIR / "index").resolve()
ADMIN_TOKEN    = os.getenv("ADMIN_TOKEN", os.getenv("ADMIN_UPLOAD_SECRET", ""))  # legacy key support

for d in (BUILTIN_DIR, UPLOADS_DIR, VECTORSTORE_DIR, MECH_KB_DIR, MECH_INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── Flask app ───────────────────────────────────────────────────────────── #
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
csrf = CSRFProtect(app)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.config["JSON_AS_ASCII"] = False  # allow UTF-8 in JSON responses
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")
app.config['WTF_CSRF_TIME_LIMIT'] = None

# ─── Dataset searcher (local table) ─────────────────────────────────────── #
_LOOKUP_FALLBACK = BASE_DIR / "database" / "tables" / "coremof.xlsx"
LOOKUP_FILE = (
    os.getenv("LOOKUP_FILE")
    or os.getenv("LOOKUP_DIR")
    or (_LOOKUP_FALLBACK if _LOOKUP_FALLBACK.exists() else "")
)

_SEARCHER: DatasetSearcher | None = None
if LOOKUP_FILE:
    try:
        _TABLE     = load_table(LOOKUP_FILE)
        _SEARCHER  = DatasetSearcher(_TABLE)
        print(f"[dataset_search] loaded '{LOOKUP_FILE}'  rows={_TABLE.shape[0]}")
    except Exception as e:
        print("[dataset_search] failed to load lookup table:", e)
else:
    print("[dataset_search] no LOOKUP_FILE set – table searching disabled.")

def basic_search(query: str, n: int = 6) -> list[dict]:
    if not query.strip():
        return []

    # ── 1) LOCAL DATASET ──────────────────────────────────────────────
    local_hits: list[dict] | pd.DataFrame
    if _SEARCHER is None:
        local_hits = []
    else:
        try:
            local_hits = _SEARCHER.query(query, topk=n)
        except Exception as e:
            print("[basic_search] local query failed:", e)
            local_hits = []

    local: list[dict] = []

    if isinstance(local_hits, pd.DataFrame):
        for _, row in local_hits.fillna("").iterrows():
            d = row.to_dict()
            for k in ("title", "year", "url", "doi"):
                d.setdefault(k, "")
            local.append(d)
    elif isinstance(local_hits, list):
        for d in local_hits:
            for k in ("title", "year", "url", "doi"):
                d.setdefault(k, "")
            local.append(d)

    # ── 2) OPENALEX (internet) ───────────────────────────────────────
    web = []
    try:
        web = search_papers(query, n)
    except Exception as e:
        print("[basic_search] OpenAlex fetch failed:", e)

    # ── 3) MERGE + DEDUP ─────────────────────────────────────────────
    seen = {(d.get("doi") or d.get("title", "")).lower() for d in local}
    for w in web:
        key = (w.get("doi") or w.get("title", "")).lower()
        if key not in seen:
            local.append({k: w.get(k, "") for k in ("title", "year", "url", "doi")})
            seen.add(key)

    return local[: 2*n]

# ─── OpenAI client (no proxy) ───────────────────────────────────────────── #
load_dotenv()
_no_proxy_client = httpx.Client(trust_env=False, timeout=120.0)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=_no_proxy_client)
app.config["OPENAI_CLIENT"] = client

# ─── Register mechanistic blueprint ─────────────────────────────────────── #
from app_extensions.mechanism_routes import mechanism_bp
app.register_blueprint(mechanism_bp)
from ingestion.ingest_mechanisms import ingest as ingest_mechanisms

# Job registry
JOBS = {}  # {job_id: {"status": "...", "progress": int, "error": str, "filename": str}}
def _set_job(jid, **kw):
    JOBS.setdefault(jid, {}).update(kw)

def _process_pdf_job(jid: str, path: Path, filename: str):
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
                {
                    "$set": {
                        "status": "indexed",
                        "indexed_at": datetime.utcnow(),
                        "n_pages": n,
                    }
                },
                upsert=True,
            )
        _set_job(jid, status="done", progress=100)
    except Exception as e:
        if db is not None:
            try:
                db.uploads.update_one(
                    {"filename": filename},
                    {
                        "$set": {
                            "status": "error",
                            "error": str(e),
                            "failed_at": datetime.utcnow(),
                        }
                    },
                    upsert=True,
                )
            except Exception as ee:
                print("[/upload] failed to record error in DB:", ee)
        _set_job(jid, status="error", error=str(e))

# ---------------- Vector Store init ----------------
def preload_builtin():
    root = os.getenv("BUILTIN_DIR") or str((Path(__file__).parent / "builtin"))
    p = Path(root)
    if not p.exists():
        print(f"[preload] no builtin dir: {p}")
        return
    count = 0
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".txt", ".md", ".json"}:
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
            if txt.strip():
                vs.add_to_store(txt, tag=f"builtin:{f.name}")
                count += 1
        except Exception as e:
            print(f"[preload] skip {f}: {e}")
    print(f"[preload] indexed {count} builtin docs from {p}")
    src = f"[SRC builtin:{f.name}]\n"
    vs.add_to_store(src + txt, tag=f"builtin:{f.name}")

try:
    if os.getenv("PRELOAD_BUILTIN", "1") == "1":
        preload_builtin()
except Exception as e:
    print("[preload] failed:", e)

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf)

@app.before_request
def _log_path():
    try:
        print(f"[req] {request.method} {request.path}")
    except Exception:
        pass

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
        return "<h1>NanoChemGPT is up</h1><p>templates/index.html is missing.</p>", 200

# --- DB context options ---
USE_DB_CONTEXT = os.getenv("USE_DB_CONTEXT", "1") == "1"
DB_CTX_LIMIT = int(os.getenv("DB_CTX_LIMIT", "3"))
DB_CTX_MAX_CHARS = int(os.getenv("DB_CTX_MAX_CHARS", "1200"))

def fetch_db_context(q: str, limit: int = DB_CTX_LIMIT) -> str:
    try:
        db = get_db()
        try:
            cur = db.qa.find({"$text": {"$search": q}})
        except Exception:
            cur = db.qa.find({"question": {"$regex": q, "$options": "i"}})
        items = list(cur.sort("created_at", -1).limit(limit))
    except Exception as e:
        print("[db_ctx] query failed:", e)
        return ""

    blobs = []
    for d in items:
        qq = (d.get("question") or "").strip()
        aa = (d.get("answer") or "").strip()
        if not aa:
            continue
        piece = f"Q: {qq}\nA: {aa}"
        if len(piece) > DB_CTX_MAX_CHARS:
            piece = piece[:DB_CTX_MAX_CHARS] + " …"
        blobs.append(piece)
    return "\n\n---\n\n".join(blobs)

def fetch_parsed_context(q: str, limit: int = 2) -> str:
    try:
        db = get_db()
        try:
            cur = db.parsed.find({"$text": {"$search": q}})
        except Exception as e:
            print("[parsed_ctx] $text unavailable, falling back to regex:", e)
            cur = db.parsed.find({
                "$or": [
                    {"question": {"$regex": q, "$options": "i"}},
                    {"raw_text": {"$regex": q, "$options": "i"}},
                ]
            })
        items = list(cur.sort("created_at", -1).limit(limit))
    except Exception as e:
        print("[parsed_ctx] query failed:", e)
        return ""

    pieces = []
    for d in items:
        p = d.get("parsed") or {}
        hdr = "; ".join(p.get("hardware", [])[:5])
        reag = "; ".join((r.get("description") for r in p.get("reagents", [])[:6] if isinstance(r, dict)))
        proc = "; ".join(p.get("procedure", [])[:6])
        parts = []
        if hdr: parts.append(f"Hardware: {hdr}")
        if reag: parts.append(f"Materials: {reag}")
        if proc: parts.append(f"Procedure: {proc}")
        if parts:
            pieces.append(" • ".join(parts))
    return "\n".join(pieces)

# --- Admin auth decorator ---
def require_admin(fn):
    @functools.wraps(fn)
    def w(*a, **kw):
        auth = request.headers.get("Authorization", "")
        token = auth.split(" ", 1)[1].strip() if auth.startswith("Bearer ") else ""
        ok = bool(ADMIN_TOKEN) and token == ADMIN_TOKEN
        if not ok:
            print(f"[admin] unauthorized: got='{token[:6]}…' expected_set={bool(ADMIN_TOKEN)} path={request.path}")
            return jsonify({"error":"unauthorized"}), 401
        return fn(*a, **kw)
    return w

def _admin_csp(resp):
    resp.headers["Content-Security-Policy"] = "default-src 'self'; connect-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:"
    return resp

# ---------------- Initialize Datasets ----------------
@app.post("/admin/upload_builtin")
@require_admin
def admin_upload_builtin():
    """
    Upload a builtin dataset file. Accepts either a plain `.json` file or a
    compressed `.json.xz` file. Saves the uploaded file into the configured
    BUILTIN_DIR and, if compressed, writes a decompressed `.json` alongside
    the original.
    """
    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"error": "no file"}), 400

    fname = secure_filename(f.filename)
    raw_path = BUILTIN_DIR / fname
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # Save original upload
    f.save(raw_path)

    # If it’s .json.xz → decompress alongside to .json
    out_path = raw_path
    if fname.lower().endswith(".json.xz"):
        out_path = BUILTIN_DIR / fname[:-3]  # strip ".xz" → keep ".json"
        with lzma.open(raw_path, "rb") as xzf, open(out_path, "wb") as out:
            out.write(xzf.read())

    return jsonify({
        "ok": True,
        "saved": str(raw_path),
        "decompressed": str(out_path) if out_path != raw_path else None
    })
    
# ---------------- Mechanistic KB Upload/Ingester --------------------------
@app.post("/admin/upload_mechanistic")
@require_admin
def admin_upload_mechanistic():
    """
    Accepts .json or .jsonl of mechanistic entries (matching schemas/mechanistic.schema.json)
    and appends them to mechanistic_kb/mechanistic.jsonl via the ingestion pipeline.
    """
    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"error": "no file"}), 400
    fname = secure_filename(f.filename)
    raw_path = (MECH_KB_DIR / fname)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    f.save(raw_path)

    # Load entries
    entries = []
    try:
        if fname.lower().endswith(".jsonl"):
            with raw_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
        elif fname.lower().endswith(".json"):
            payload = json.loads(raw_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload, list):
                entries.extend(payload)
            elif isinstance(payload, dict):
                entries.append(payload)
            else:
                return jsonify({"error": "JSON must be an object or an array of objects"}), 400
        else:
            return jsonify({"error": "Unsupported file type (use .json or .jsonl)"}), 400
    except Exception as e:
        return jsonify({"error": f"failed to parse file: {e}"}), 400

    try:
        ids = ingest_mechanisms(entries)
        return jsonify({"ok": True, "ingested": len(ids), "ids": ids})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"ingestion failed: {e}"}), 500

# ---------------- Upload ----------------
@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        abort(400, "No file uploaded.")
    fname = secure_filename(f.filename)
    path = UPLOADS_DIR / fname
    f.save(path)

    try:
        db = get_db()
        db.uploads.update_one(
            {"filename": fname},
            {"$set": {"filename": fname, "ts": datetime.utcnow(), "status": "received"}},
            upsert=True,
        )
    except Exception as e:
        print("[/upload] DB receipt warn:", e)

    jid = os.urandom(8).hex()
    _set_job(jid, status="processing", progress=0, filename=fname)

    try:
        lower = fname.lower()
        if lower.endswith(".pdf"):
            threading.Thread(target=_process_pdf_job, args=(jid, path, fname), daemon=True).start()
        elif lower.endswith(".json"):
            raw = path.read_text(encoding="utf-8", errors="ignore")
            vs.add_to_store(raw, tag=f"upload:{fname}")
            try:
                get_db().uploads.update_one(
                    {"filename": fname},
                    {"$set": {"status": "indexed", "indexed_at": datetime.utcnow(), "kind": "json"}},
                    upsert=True,
                )
            except Exception as e:
                print("[/upload] DB update warn:", e)
            _set_job(jid, status="done", progress=100)
        else:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            vs.add_to_store(txt, tag=f"upload:{fname}")
            try:
                get_db().uploads.update_one(
                    {"filename": fname},
                    {"$set": {"status": "indexed", "indexed_at": datetime.utcnow(), "kind": "text"}},
                    upsert=True,
                )
            except Exception as e:
                print("[/upload] DB update warn:", e)
            _set_job(jid, status="done", progress=100)
    except Exception as e:
        _set_job(jid, status="error", error=str(e))

    return jsonify({"ok": True, "job_id": jid, "filename": fname})

@app.get("/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    if not j:
        abort(404, "unknown job id")
    return jsonify(j)

# ---------------- Search ----------------
@app.post("/search")
def search_route():
    payload = request.get_json(silent=True) or {}
    q = (payload.get("q") or payload.get("query") or "").strip()
    n = int(payload.get("n") or 6)
    if not q:
        abort(400, "Missing 'q' (query).")
    refs = basic_search(q, n)
    return jsonify({"results": refs})

# ---------------- Ask ----------------

_CIT_BRACKET_RX = re.compile(r"[\[](?P<num>\d{1,4})\]")
_CIT_FULLWIDTH_RX = re.compile(r"【(?P<num>\d{1,4})】")
_CIT_FOOTNOTE_RX = re.compile(r"\[\^(?P<num>\d{1,4})\]")
_TAGS = ("CTX", "PARSED", "DB", "GEN")

def _extract_used_markers(*texts: str) -> dict:
    seen = set()
    tag_counts = {t: 0 for t in _TAGS}
    for t in texts:
        if not t:
            continue
        tt = t.replace('\u00A0', ' ')
        for rx in (_CIT_BRACKET_RX, _CIT_FULLWIDTH_RX, _CIT_FOOTNOTE_RX):
            for m in rx.finditer(tt):
                try:
                    seen.add(int(m.group('num')))
                except Exception:
                    pass
        for tag in _TAGS:
            tag_rx = re.compile(rf"\[{tag}\]")
            tag_counts[tag] += len(tag_rx.findall(tt))
    refs = sorted(seen)
    has_ctx = any(tag_counts[t] > 0 for t in ("CTX", "PARSED", "DB"))
    return {"refs": refs, "tags": tag_counts, "has_ctx": has_ctx}

@app.post("/ask")
def ask():
    def split_reasoning(raw: str) -> tuple[str, str]:
        if not raw:
            return "", ""
        text = raw.strip()
        fence = re.compile(r"```(?:reason|rationale|reasoning)\s*(.*?)```", re.I | re.S)
        m = fence.search(text)
        if m:
            rationale = m.group(1).strip()
            answer = (text[:m.start()] + text[m.end():]).strip()
            return answer, rationale
        fence_any = re.compile(r"rationale\s*:?\s*```(.*?)```", re.I | re.S)
        m = fence_any.search(text)
        if m:
            rationale = m.group(1).strip()
            answer = (text[:m.start()] + text[m.end():]).strip()
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
        if not q:
            abort(400, "No question.")

        vs_ctx = ""
        try:
            vs_ctx = vs.search(q, k=8) or ""
        except Exception as e:
            print("[/ask] vs.search error:", e)
            vs_ctx = ""

        db_ctx = ""  # disabled: do not use DB Q&A context
        ctx_parsed = ""  # disabled: do not use parsed DB context

        context_parts = []
        if vs_ctx: 
            context_parts.append("<<<CTX_UPLOADS>>>\n" + vs_ctx)
            context_joined = "\n\n---\n\n".join(context_parts).strip()
            print("[ask][debug] VS tags preview:", (vs_ctx or "").replace("\n"," ")[:220])
            print("[ask] ctx parts:", f"VS={bool(vs_ctx)} len={len(vs_ctx) if vs_ctx else 0}")

        refs = []
        try:
            refs = basic_search(q, n=6) or []
        except Exception as e:
            print("[/ask] basic_search error:", e)
            refs = []

        def _ref_url(r):
            if r.get("url"): return r["url"]
            if r.get("doi"): return f"https://doi.org/{r['doi']}"
            return ""

        refs_prompt = "\n".join(
            f"[{i+1}] {(r.get('title') or '(no title)')} ({r.get('year') or ''}) — {_ref_url(r)}"
            for i, r in enumerate(refs)
        )

        prompt = (
            "You are NanoChemGPT. Use the CONTEXT and the numbered REFERENCES "
            "to propose a synthesis.\n"
            "Rules:\n"
            " - Prefer CONTEXT over general knowledge when relevant.\n"
            " - If you use any content from CONTEXT, append [CTX] on that line.\n"
            " - When you pull a fact from any numbered REFERENCE, put its number in "
            "   square brackets right after the sentence (e.g. “hydrothermal at 200 °C [3]”).\n"
            " - If CONTEXT is insufficient, say so explicitly before generalizing.\n"
            "Return two blocks exactly in this order:\n"
            "## SynthesisProtocol\n"
            "1. **Hardware & Glassware**:\n[]\n"
            "2. **Materials**:\n[]\n"
            "3. **Procedure**\n[]\n\n"
            "```reason\n"
            "For each key justification, add inline tags: [CTX] for uploaded/context hits, "
            "[DB] for Mongo Q&A, [PARSED] for parsed protocols, [n] for numbered web REFERENCES, [GEN] if inferred.\n"
            "Keep rationales terse.\n"
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

        answer, rationale = split_reasoning(raw)

        if not (rationale or "").strip():
            try:
                rationale_only = (
                    "You previously produced the SynthesisProtocol below.\n"
                    "Write a short rationale (5–8 bullets max). For each key justification add inline tags:\n"
                    "[CTX] uploaded/context hits, [DB] Mongo Q&A, [PARSED] parsed protocols, [n] for numbered web REFERENCES, [GEN] for general.\n"
                    "Base your reasoning strictly on the ANSWER, CONTEXT, and REFERENCES.\n"
                    "Return just the rationale text, no code fences, no extra headings."
                )
                rraw = client.chat.completions.create(
                    model="gpt-4o-mini",
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

        try:
            used_summary = _extract_used_markers(answer or "", rationale or "")
            if not used_summary.get("refs"):
                try:
                    revise_refs = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        messages=[
                            {"role": "system", "content": (
                                "Add inline [n] citations wherever you used information "
                                "from the numbered REFERENCES list. Do NOT remove any "
                                "existing [CTX] or other content. Only insert citations "
                                "where appropriate."
                            )},
                            {"role": "user",   "content": f"REFERENCES:\n{refs_prompt}"},
                            {"role": "user",   "content": f"ORIGINAL ANSWER:\n{answer}"}
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
            
        if not used_summary.get("has_ctx"):
            try:
                revise = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                    {"role":"system","content":"Revise the answer to explicitly use CONTEXT where relevant. Insert [CTX] markers on lines that derive from CONTEXT, and prefer CONTEXT over general knowledge. Do not change structure."},
                    {"role":"user","content": f"CONTEXT:\n{context_joined}"},
                    {"role":"user","content": f"ORIGINAL ANSWER:\n{answer}"}
                    ]
                ).choices[0].message.content
                if revise and len(revise) >= 0.7 * len(answer):
                    answer = revise
                    used_summary = _extract_used_markers(answer, rationale or "")
            except Exception as e:
                print("[ask] revise step skipped:", e)

        qa_id = None
        try:
            db = get_db()
            ins = db.qa.insert_one({
                "created_at": datetime.utcnow(),
                "question": q,
                "answer": (answer or "").strip(),
                "rationale": rationale,
                "references": refs,
                "refs_used": used_summary.get("refs", []),
                "used_tags": used_summary.get("tags", {}),
                "ctx_vs": vs_ctx,
                "ctx_db": db_ctx,
                "ctx_parsed": ctx_parsed,
            })
            qa_id = str(ins.inserted_id)
        except Exception as e:
            print("[/ask] DB insert warn:", e)

        return jsonify({
            "answer": (answer or "").strip(),
            "rationale": rationale,
            "references": refs,
            "refs_used": used_summary.get("refs", []),
            "used": used_summary,
            "qa_id": qa_id,
            "ctx_vs": (vs_ctx or "")[:8000],        })

    except Exception as e:
        print("[/ask] Unhandled error:", e)
        traceback.print_exc()
        return jsonify({"error": f"/ask failed: {e}"}), 500

# ---------------- Parse & Save ----------------
@app.post("/parse")
def parse_route():
    """
    New parser endpoint using the validator/normalizer in converter.py.

    Request JSON:
      { "text": "<dict or Key: value lines>" }

    Response JSON:
      { "ok": true, "data": <normalized & validated dict> }
    """
    try:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "JSON must contain non-empty 'text'"}), 400

        # validate_step accepts either dict-like JSON (string) or Key: value text.
        data = validate_step(text)
        return jsonify({"ok": True, "data": data})

    except ValueError as ve:
        # Raised by validate_step with line-aware messages
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
    """
    Accept a .txt or .md file containing a human-edited 'Robot mode' answer,
    run it through the existing converter.validate_step(), and return JSON.
    """
    try:
        f = request.files.get("file")
        if not f or f.filename == "":
            return jsonify({"ok": False, "error": "no file"}), 400

        # read as utf-8 text (tolerant)
        try:
            text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            # werkzeug FileStorage may already be str with .read() returning str on some stacks
            text = f.read()

        if not (text or "").strip():
            return jsonify({"ok": False, "error": "file is empty"}), 400

        data = validate_step(text)
        return jsonify({"ok": True, "data": data})
    except ValueError as ve:
        # raised by validate_step with line-aware messages
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
@require_admin
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

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("DEBUG", "0") == "1"
    )
