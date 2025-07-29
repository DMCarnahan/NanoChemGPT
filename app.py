import os, io, json, threading, traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, abort, render_template, send_file
from jinja2 import TemplateNotFound

# Local imports
import vector_store as vs
from converter import convert_to_json, ParserError
from search import basic_search

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

@app.get("/health")
def health():
    return "ok", 200

@app.get("/")
def home():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return "<h1>NanoChemGPT is up</h1><p>templates/index.html is missing.</p>", 200

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        abort(400, "No file uploaded.")
    fname = secure_filename(f.filename)
    path = UPLOADS_DIR / fname
    f.save(path)

    jid = os.urandom(8).hex()
    _set_job(jid, status="processing", progress=0, filename=fname)

    try:
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
    except Exception as e:
        _set_job(jid, status="error", error=str(e))

    return jsonify({"ok": True, "job_id": jid, "filename": fname})

@app.get("/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    if not j:
        abort(404, "unknown job id")
    return jsonify(j)

@app.post("/search")
def search_route():
    payload = request.get_json(silent=True) or {}
    q = (payload.get("q") or payload.get("query") or "").strip()
    n = int(payload.get("n") or 6)
    if not q:
        abort(400, "Missing 'q' (query).")
    refs = basic_search(q, n)
    return jsonify({"results": refs})

@app.post("/ask")
def ask():
    try:
        payload = request.get_json(silent=True) or {}
        q = (payload.get("question") or "").strip()
        if not q:
            abort(400, "No question.")

        try:
            context = vs.search(q, k=4)
        except Exception as e:
            print("[/ask] vs.search error:", e); context = ""

        refs = []
        try:
            refs = basic_search(q, n=6) or []
        except Exception as e:
            print("[/ask] basic_search error:", e); refs = []

        def _ref_url(r):
            if r.get("url"): return r["url"]
            if r.get("doi"): return f"https://doi.org/{r['doi']}"
            return ""
        refs_prompt = "\n".join(
            f"[{i+1}] {(r.get('title') or '(no title)')} ({r.get('year') or ''}) — {_ref_url(r)}"
            for i, r in enumerate(refs)
        )

        prompt = (
        "You are NanoChemGPT, an AI assistant that proposes nanomaterial syntheses.\n"
        "Use the provided context unless general chemistry knowledge is required.\n"
        "Provide concrete numerical parameters on the same volume scale as the paper.\n\n"
        "Return *two blocks* in order:\n"
        "## SynthesisProtocol\n"
        "1. **Hardware & Glassware**:\n[]\n"
        "2. **Materials**:\n[]\n"
        "3. **Procedure**\n[]\n\n"
        "```reason\n"
        "CITATION RULES (very important):\n"
        "• In the rationale, add inline numeric citations like [1], [2] at the end of ANY sentence that relies on literature.\n"
        "• Use ONLY the numbers from the 'Retrieved sources' list below. Do NOT invent new numbers.\n"
        "• If a sentence is based on the provided vector-store context (not public web), use [CTX].\n"
        "• If it is general chemistry knowledge with no specific citation, use [GEN].\n"
        "• Prefer 1–2 citations per sentence (avoid over-citation).\n\n"
        "Think step-by-step:\n"
        "1) Restate constraints. 2) Justify every solvent/ratio/temp. 3) Final-check for violations.\n" \
        "Style constraints for Procedure:\n"
        "- Use imperative, atomic steps.\n"
        "- No explanatory prose inside steps.\n"
        "- Put explanatory sentences only in the ```reason block, with citations.\n"
            f"\n\nCONTEXT:\n{context}\n\nREFERENCES:\n{refs_prompt}\n\nUser question: {q}"
        )

        raw = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ).choices[0].message.content

        if "```reason" in raw:
            answer, rest = raw.split("```reason", 1)
            rationale = rest.split("```", 1)[0].strip()
        else:
            answer, rationale = raw, ""

        return jsonify({"answer": answer.strip(), "rationale": rationale, "references": refs})
    except Exception as e:
        print("[/ask] Unhandled error:", e)
        traceback.print_exc()
        return jsonify({"error": f"/ask failed: {e}"}), 500

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
    return jsonify(parsed)

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

@app.post("/clear_uploads")
def clear_uploads_route():
    try:
        vs.clear_uploads()
    except Exception as e:
        print("clear_uploads error:", e)
    return {"status": "uploads cleared"}

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