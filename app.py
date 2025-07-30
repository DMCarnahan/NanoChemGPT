import os, io, json, threading, traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, abort, render_template, send_file
from jinja2 import TemplateNotFound
from bson import ObjectId

# Local imports 
import vector_store as vs
from converter import convert_to_json, ParserError
from search import basic_search
from mongo_client import get_db, ping as mongo_ping

# App + folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__,
            template_folder=str(BASE_DIR / "templates"),
            static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Job registry
JOBS = {}  # {job_id: {"status": "...", "progress": int, "error": str, "filename": str}}
def _set_job(jid, **kw): JOBS.setdefault(jid, {}).update(kw)

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
            db.uploads.update_one({"filename": filename},
                                  {"$set": {"status": "indexed", "indexed_at": datetime.utcnow(),
                                            "n_pages": n}},
                                  upsert=True)
        _set_job(jid, status="done", progress=100)
    except Exception as e:
        if db is not None:
            try:
                db.uploads.update_one({"filename": filename},
                                      {"$set": {"status": "error", "error": str(e),
                                                "failed_at": datetime.utcnow()}},
                                      upsert=True)
            except Exception as ee:
                print("[/upload] failed to record error in DB:", ee)
        _set_job(jid, status="error", error=str(e))

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
USE_DB_CONTEXT   = os.getenv("USE_DB_CONTEXT", "1") == "1"
DB_CTX_LIMIT     = int(os.getenv("DB_CTX_LIMIT", "3"))
DB_CTX_MAX_CHARS = int(os.getenv("DB_CTX_MAX_CHARS", "1200"))

def fetch_db_context(q: str, limit: int = DB_CTX_LIMIT) -> str:
    """Return recent similar Q&A from Mongo (db.qa) as a compact text block."""
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
    """Return succinct summaries built from parsed protocols in db.parsed."""
    try:
        db = get_db()
        cur = db.parsed.find({"raw_text": {"$regex": q, "$options": "i"}}).sort("created_at", -1).limit(limit)
        items = list(cur)
    except Exception as e:
        print("[parsed_ctx] query failed:", e)
        return ""

    pieces = []
    for d in items:
        p = d.get("parsed") or {}
        hdr  = "; ".join(p.get("hardware", [])[:5])
        reag = "; ".join((r.get("description") for r in p.get("reagents", [])[:6] if isinstance(r, dict)))
        proc = "; ".join(p.get("procedure", [])[:6])
        parts = []
        if hdr:  parts.append(f"Hardware: {hdr}")
        if reag: parts.append(f"Materials: {reag}")
        if proc: parts.append(f"Procedure: {proc}")
        if parts:
            pieces.append(" • ".join(parts))
    return "\n\n".join(pieces)

# ---------------- Upload ----------------
@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        abort(400, "No file uploaded.")
    fname = secure_filename(f.filename)
    path = UPLOADS_DIR / fname
    f.save(path)

    # record receipt
    try:
        db = get_db()
        db.uploads.update_one({"filename": fname},
                              {"$set": {"filename": fname, "ts": datetime.utcnow(), "status": "received"}},
                              upsert=True)
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
                get_db().uploads.update_one({"filename": fname},
                                            {"$set": {"status": "indexed", "indexed_at": datetime.utcnow(),
                                                      "kind": "json"}},
                                            upsert=True)
            except Exception as e:
                print("[/upload] DB update warn:", e)
            _set_job(jid, status="done", progress=100)
        else:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            vs.add_to_store(txt, tag=f"upload:{fname}")
            try:
                get_db().uploads.update_one({"filename": fname},
                                            {"$set": {"status": "indexed", "indexed_at": datetime.utcnow(),
                                                      "kind": "text"}},
                                            upsert=True)
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
@app.post("/ask")
def ask():
    from datetime import datetime
    import re, traceback

    def split_reasoning(raw: str) -> tuple[str, str]:
        """Extract rationale robustly from fences or headings."""
        if not raw:
            return "", ""
        text = raw.strip()

        # 1) Fenced code block with language: reason / rationale / reasoning
        fence = re.compile(r"```(?:reason|rationale|reasoning)\s*(.*?)```",
                           re.I | re.S)
        m = fence.search(text)
        if m:
            rationale = m.group(1).strip()
            answer = (text[:m.start()] + text[m.end():]).strip()
            return answer, rationale

        # 2) Any fenced block after a 'rationale' keyword in the line above
        #    e.g., "... rationale:\n``` ... ```"
        fence_any = re.compile(r"rationale\s*:?\s*```(.*?)```", re.I | re.S)
        m = fence_any.search(text)
        if m:
            rationale = m.group(1).strip()
            answer = (text[:m.start()] + text[m.end():]).strip()
            return answer, rationale

        # 3) Markdown heading "Rationale" / "Reasoning"
        head = re.compile(r"(?:^|\n)#{1,3}\s*(rationale|reasoning)\b[^\n]*\n(.*)$",
                          re.I | re.S)
        m = head.search(text)
        if m:
            rationale = m.group(2).strip()
            answer = text[:m.start()].strip()
            return answer, rationale

        # 4) Fallback: no rationale
        return text, ""

    try:
        payload = request.get_json(silent=True) or {}
        q = (payload.get("question") or "").strip()
        if not q:
            abort(400, "No question.")

        # --- Build context from vector store first ---
        vs_ctx = ""
        try:
            vs_ctx = vs.search(q, k=4) or ""
        except Exception as e:
            print("[/ask] vs.search error:", e)

        # --- Optional Mongo context (history + parsed) ---
        db_ctx = fetch_db_context(q) if USE_DB_CONTEXT else ""
        ctx_parsed = fetch_parsed_context(q) if USE_DB_CONTEXT else ""

        # Order: VS first, PARSED second, DB last (most general).
        context_parts = []
        if vs_ctx:     context_parts.append("<<<CTX_UPLOADS>>>\n" + vs_ctx)
        if ctx_parsed: context_parts.append("<<<CTX_PARSED>>>\n"  + ctx_parsed)
        if db_ctx:     context_parts.append("<<<CTX_DB_QA>>>\n"   + db_ctx)

        context_joined = "\n\n---\n\n".join(context_parts).strip()

        # --- Web refs for citations ---
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

        # --- Prompt ---
        prompt = (
            "You are NanoChemGPT. Use the CONTEXT and the numbered REFERENCES to propose a synthesis.\n"
            "Return two blocks exactly in this order:\n"
            "## SynthesisProtocol\n"
            "1. **Hardware & Glassware**:\n[]\n"
            "2. **Materials**:\n[]\n"
            "3. **Procedure**\n[]\n\n"
            "```reason\n"
            "For each key justification, add an inline tag:\n"
            "  [CTX] for uploaded/context hits, [DB] for similar past Q&A from Mongo,\n"
            "  [PARSED] for summaries from prior parsed protocols,\n"
            "  [n] for numbered web REFERENCES, and [GEN] if inferred/general.\n"
            "Keep rationales terse.\n"
            "```\n\n"
            f"CONTEXT:\n{context_joined}\n\n"
            f"REFERENCES:\n{refs_prompt}\n\n"
            f"User question: {q}"
        )

        raw = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ).choices[0].message.content

        answer, rationale = split_reasoning(raw)

        # --- Save to Mongo (includes exact context parts for auditing) ---
        qa_id = None
        try:
            db = get_db()
            ins = db.qa.insert_one({
                "created_at": datetime.utcnow(),
                "question": q,
                "answer": (answer or "").strip(),
                "rationale": rationale,
                "references": refs,
                "ctx_vs": vs_ctx,
                "ctx_db": db_ctx,
                "ctx_parsed": ctx_parsed,
            })
            qa_id = str(ins.inserted_id)
        except Exception as e:
            print("[/ask] DB insert warn:", e)

        # --- Return everything the UI needs (including ctx fields) ---
        return jsonify({
            "answer": (answer or "").strip(),
            "rationale": rationale,
            "references": refs,
            "qa_id": qa_id,
            "ctx_vs": (vs_ctx or "")[:8000],
            "ctx_parsed": (ctx_parsed or "")[:8000],
            "ctx_db": (db_ctx or "")[:8000],
        })

    except Exception as e:
        print("[/ask] Unhandled error:", e)
        traceback.print_exc()
        return jsonify({"error": f"/ask failed: {e}"}), 500

# ---------------- Parse ----------------
@app.post("/parse")
def parse_route():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        abort(400, "JSON must contain non‑empty 'text'.")
    robot = bool(payload.get("robot"))
    try:
        parsed = convert_to_json(text, robot=robot)
    except ParserError as e:
        abort(422, str(e))
    # Save to Mongo
    try:
        db = get_db()
        db.parsed.insert_one({
            "created_at": datetime.utcnow(),
            "robot": robot,
            "raw_text": text,
            "parsed": parsed
        })
    except Exception as e:
        print("[/parse] DB insert warn:", e)
    return jsonify(parsed)

# ---------------- Save TXT ----------------
@app.post("/save_txt")
def save_txt():
    data = request.get_json(silent=True) or {}
    answer   = (data.get("answer") or "").strip()
    question = (data.get("question") or "").strip()
    if not answer:
        abort(400, "answer is empty")
    buf = io.BytesIO(f"Q: {question}\n\nA:\n{answer}\n".encode())
    buf.seek(0)
    fname = f"chatau_{datetime.utcnow():%Y%m%d_%H%M%S}.txt"
    return send_file(buf, mimetype="text/plain", as_attachment=True, download_name=fname)

# ---------------- Upload maintenance ----------------
@app.post("/clear_uploads")
def clear_uploads_route():
    try:
        vs.clear_uploads()
    except Exception as e:
        print("clear_uploads error:", e)
    return {"status": "uploads cleared"}

# ---------------- History & Upload Browser APIs ----------------
def _safe_id(x):
    try:
        return ObjectId(x)
    except Exception:
        return None

def _doc(obj):
    if not isinstance(obj, dict): return obj
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
    items = [ _doc(d) for d in cur.sort("created_at", -1).skip(skip).limit(limit) ]
    return jsonify({"items": items, "skip": skip, "limit": limit})

@app.get("/api/history/<id>")
def api_history_one(id):
    db = get_db()
    oid = _safe_id(id)
    if not oid: abort(404, "invalid id")
    doc = db.qa.find_one({"_id": oid})
    if not doc: abort(404, "not found")
    return jsonify(_doc(doc))

@app.get("/api/uploads")
def api_uploads():
    db = get_db()
    try:
        limit = min(200, int(request.args.get("limit", 50)))
    except Exception:
        limit = 50
    cur = db.uploads.find({}).sort([("indexed_at", -1), ("ts", -1)]).limit(limit)
    items = [ _doc(d) for d in cur ]
    return jsonify({"items": items, "limit": limit})

# ---------------- Errors ----------------
@app.errorhandler(400)
@app.errorhandler(422)
@app.errorhandler(500)
def handle_err(e):
    return jsonify(error=str(e)), getattr(e, "code", 500)

@app.errorhandler(413)
def too_large(e):
    return jsonify(error="File bigger than 100 MB — compress or split it."), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=bool(os.getenv("DEBUG")))
