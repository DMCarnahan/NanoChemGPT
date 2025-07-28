import os, io, json, threading, uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, abort, render_template, send_file
from jinja2 import TemplateNotFound

# === project-local imports ========================
import vector_store as vs                         
from parser import convert_to_json, ParserError  
# ==========================================================================

# ---- paths & app ----------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Limit request body size (matches 413 handler below)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- small helpers --------------------------------------------------------
def _extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# In‑memory job registry 
JOBS = {}  # {job_id: {"status": "...", "progress": int, "error": str, "filename": str}}

def _set_job(jid, **kw):
    JOBS.setdefault(jid, {}).update(kw)

def _process_pdf_job(jid: str, path: Path, filename: str):
    """Background thread that extracts text from a PDF and adds it to the vector store."""
    try:
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
        _set_job(jid, status="done", progress=100)
    except Exception as e:
        _set_job(jid, status="error", error=str(e))

# ---- routes: health & home ------------------------------------------------
@app.get("/health")
def health():
    return "ok", 200

@app.get("/")
def home():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return "<h1>NanoChemGPT is up</h1><p>templates/index.html is missing.</p>", 200

# ---- routes: upload + status (non‑blocking) -------------------------------
@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        abort(400, "No file uploaded.")

    fname = secure_filename(f.filename)
    path = UPLOADS_DIR / fname
    f.save(path)

    jid = uuid.uuid4().hex
    _set_job(jid, status="processing", progress=0, filename=fname)

    lower = fname.lower()
    if lower.endswith(".pdf"):
        threading.Thread(target=_process_pdf_job, args=(jid, path, fname), daemon=True).start()
    elif lower.endswith(".json"):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        vs.add_to_store(raw, tag=f"upload:{fname}")
        _set_job(jid, status="done", progress=100)
    else:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        vs.add_to_store(txt, tag=f"upload:{fname}")
        _set_job(jid, status="done", progress=100)

    return jsonify({"ok": True, "job_id": jid, "filename": fname})

@app.get("/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    if not j:
        abort(404, "unknown job id")
    return jsonify(j)

# ---- routes: main Q&A -----------------------------------------------------
@app.post("/ask")
def ask():
    q = (request.form.get("question") or request.json.get("question") if request.is_json else "").strip() \
        if request.method == "POST" else ""
    if not q:
        abort(400, "No question.")

    # Retrieve context from vector store
    ctx = vs.search(q, k=4)
    if isinstance(ctx, (list, tuple)):
        context = "\n\n".join(map(str, ctx))
    else:
        context = str(ctx)

    prompt = (
        "You are NanoChemGPT, an AI assistant that proposes nanomaterial syntheses. "
        "Use the context unless general chemistry knowledge is required. "
        "Provide concrete numerical parameters on the same volume scale as the paper. "
        "Return *two blocks* in order:\n"
        "## SynthesisProtocol\n"
        "1. **Hardware & Glassware**:\n[]\n"
        "2. **Materials**:\n[]\n"
        "3. **Procedure**\n[]\n\n"
        "```reason\n"
        "Think step‑by‑step:\n"
        "1. Restate constraints.\n"
        "2. Justify every solvent / ratio / temp.\n"
        "3. Final‑check for violations (e.g. water in air-free reaction → reject).\n"
        f"Context:\n{context}\n\nUser question: {q}\n```"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""

    if "```reason" in raw:
        answer, rest = raw.split("```reason", 1)
        rationale = rest.split("```", 1)[0].strip()
    else:
        answer, rationale = raw, ""

    return {"answer": answer.strip(), "rationale": rationale}

# ---- routes: JSON parse ---------------------------------------------------
@app.post("/parse")
def parse_route():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        abort(400, "JSON must contain non‑empty 'text'.")
    try:
        parsed = convert_to_json(text)
    except ParserError as e:
        abort(422, str(e))
    return jsonify(parsed)

# ---- routes: save answer/rationale to .txt -------------------------------
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

# ---- routes: clear uploaded vectors --------------------------------------
@app.post("/clear_uploads")
def clear_uploads_route():
    try:
        vs.clear_uploads()
    except AttributeError:
        abort(501, "clear_uploads() not implemented in vector_store.")
    return {"status": "uploads cleared"}

# ---- error handlers -------------------------------------------------------
@app.errorhandler(400)
@app.errorhandler(422)
@app.errorhandler(500)
def handle_err(e):
    return jsonify(error=str(e)), getattr(e, "code", 500)

@app.errorhandler(413)
def too_large(e):
    return jsonify(error="File bigger than 100 MB — compress or split it."), 413

# ---- dev entry ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=bool(os.getenv("DEBUG")))
