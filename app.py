"""Flask API for NanoChemGPT
--------------------------------------------------
* Upload PDF or JSON → vector_store.add_(…) (tag="upload")
* `/ask` → retrieves context (k=4) + calls GPT‑4o‑mini
* `/clear_uploads` drops all *uploaded* vectors (builtin corpus kept)
* Auto‑expiry of upload vectors handled inside vector_store.search()
"""
from __future__ import annotations

import io, os, json, gzip
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import (
    Flask, jsonify, render_template, request, abort, send_file
)
from openai import OpenAI
from PyPDF2 import PdfReader
import requests, ijson

import vector_store as vs
from backend.parser import convert_to_json, ParserError

# ──────────────────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

# limit request body to 100 MB (PDF / JSON uploads)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# ── helpers --------------------------------------------------------------

def _extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# ── routes ---------------------------------------------------------------

@app.get("/")
def home():
    return render_template("index.html")

# ---- file upload (PDF / JSON) ------------------------------------------
# upload route
if file.filename.lower().endswith(".pdf"):
    try:
        pages = PdfReader(file).pages          # avoids full read into RAM
        text  = "\n".join(p.extract_text() or "" for p in pages)
    except Exception as err:
        abort(400, f"PDF parse error: {err}")
    vs.add_to_store(text)

@app.post("/upload")
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        abort(400, "No file uploaded.")

    if file.mimetype == "application/pdf":
        vs.add_to_store(_extract_text(file.read()))

    elif file.mimetype in {"application/json", "application/x-json"}:
        vs.add_json_bytes(file.read())

    else:
        abort(400, "Only PDF or JSON accepted.")

    return {"status": "ok", "filename": file.filename}

# ---- main Q&A -----------------------------------------------------------
@app.post("/ask")
def ask():
    q = request.form.get("question", "").strip()
    if not q:
        abort(400, "No question.")

    context = vs.search(q, k=4)
    prompt  = (
        "You are NanoChemGPT, an AI assistant that proposes nanomaterial syntheses. "
        "Use the context unless general chemistry knowledge is required. "
        "Provide concrete numerical parameters on the same volume scale as the paper. "
        "Return *two blocks* in order:\n"
        "## SynthesisProtocol\n"
        "1. **Hardware**:\n[]\n"
        "2. **Materials**:\n[]\n"
        "3. **Procedure**\n[]\n\n"
        "```reason"
            "Think step‑by‑step:\n"
            "1. Restate constraints.\n"
            "2. Justify every solvent / ratio / temp.\n"
            "3. Final‑check for violations (e.g. water in air-free reaction → reject).\n"
        f"Context:\n{context}\n\nUser question: {q}"
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

    return {"answer": answer.strip(), "rationale": rationale}

# ---- JSON → structured ---------------------------------------------------
@app.post("/parse")
def parse_route():
    payload: dict[str, str] = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    if not text:
        abort(400, "JSON must contain non‑empty 'text'.")
    try:
        parsed = convert_to_json(text)
    except ParserError as e:
        abort(422, str(e))
    return jsonify(parsed)

# ---- Save answer / rationale to .txt ------------------------------------
@app.post("/save_txt")
def save_txt():
    data = request.get_json(silent=True) or {}
    answer   = data.get("answer", "").strip()
    question = data.get("question", "").strip()
    if not answer:
        abort(400, "answer is empty")

    buf = io.BytesIO(f"Q: {question}\n\nA:\n{answer}\n".encode())
    buf.seek(0)
    fname = f"chatau_{datetime.utcnow():%Y%m%d_%H%M%S}.txt"
    return send_file(buf, mimetype="text/plain", as_attachment=True, download_name=fname)

# ---- Purge all uploaded vectors -----------------------------------------
@app.post("/clear_uploads")
def clear_uploads_route():
    vs.clear_uploads()
    return {"status": "uploads cleared"}

# ---- health --------------------------------------------------------------
@app.get("/ping")
def ping():
    return {"status": "alive"}

# ---- error handler -------------------------------------------------------
@app.errorhandler(400)
@app.errorhandler(422)
@app.errorhandler(500)
def handle_err(e):
    return jsonify(error=str(e)), getattr(e, "code", 500)
@app.errorhandler(413)
def too_large(e):
    return jsonify(error="File bigger than 100 MB — compress or split it."), 413
    
# ---- dev entry -----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
