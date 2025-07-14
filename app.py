# app.py – streamlined Flask backend with rationale support
from __future__ import annotations

import io, os
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from flask import (
    Flask, jsonify, render_template, request, abort, send_file
)
from openai import OpenAI
from PyPDF2 import PdfReader

import vector_store as vs
from backend.parser import convert_to_json, ParserError

# ── init -------------------------------------------------------------------
load_dotenv()
app    = Flask(__name__, template_folder="templates", static_folder="static")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ensure nltk punkt is found (already downloaded in build stage)
#import nltk  # noqa: E402
#nltk.data.path.append("/opt/render/nltk_data")

# ── utils ------------------------------------------------------------------

def _extract_text(pdf_bytes: bytes) -> str:
    """Return full text of all pages concatenated."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# ── routes -----------------------------------------------------------------

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/upload")
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        abort(400, "No file.")
    if file.mimetype == "application/pdf":
        vs.add_to_store(_extract_text(file.read()))
    elif file.mimetype == "application/json":
        vs.add_json_to_store(file.read())
    else:
        abort(400, "Only PDF or JSON accepted.")
    return {"status": "ok", "filename": file.filename}

@app.post("/ask")
def ask():
    q = request.form.get("question", "").strip()
    if not q:
        abort(400, "No question.")

    context = vs.search(q, k=4)
    prompt  = (
        "You are NanoChemGPT, an AI assistant that proposes nanomaterial syntheses. "
        "Use the context unless general chemistry knowledge is required. "
        "Provide concrete numerical parameters on the same volume scale as the paper. You should alway think about your response before submitting. \n\n"
        "Return **two blocks** in order:\n"
        "## SynthesisProtocol\n"
        "1. **Materials**: \n[]\n"
        "2. **Procedure**\n[]\n"
        "3. **Characterization**:\n[]\n\n"
        "```reason\nExplain, step‑by‑step, why each chemical, ratio, temperature, and other parameter was chosen.\n```\n\n"
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

    return jsonify(answer=answer.strip(), rationale=rationale)


@app.post("/parse")
def parse_route():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    if not text:
        abort(400, "JSON must contain non-empty 'text'.")
    try:
        parsed = convert_to_json(text)
    except ParserError as e:
        abort(422, str(e))
    return jsonify(parsed)

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

# ── optional one-time seeding route ───────────────────────────────────────
import pathlib
SEED_FOLDER = pathlib.Path("seed_pdfs")   # put your reference PDFs here

@app.post("/seed")
def seed():
    """Ingest every PDF inside seed_pdfs/ and add it to the vector store.
       Call once, then remove or protect with SEED_TOKEN."""
    token = request.headers.get("X-SEED-TOKEN")
    if token != os.getenv("SEED_TOKEN"):          # simple gate
        abort(403)

    pdfs = list(SEED_FOLDER.glob("*.pdf"))
    if not pdfs:
        return {"status": "no PDFs found", "folder": str(SEED_FOLDER)}

    for pdf in pdfs:
        with pdf.open("rb") as f:
            vs.add_to_store(_extract_text(f.read()))
        print("seeded", pdf.name)

    return {"status": "seeded", "files": len(pdfs)}

@app.get("/ping")
def ping():
    return {"status": "alive"}

# ── JSON error handler -----------------------------------------------------
@app.errorhandler(400)
@app.errorhandler(422)
@app.errorhandler(502)
@app.errorhandler(500)
def handle_err(e):
    return jsonify(error=str(e)), getattr(e, "code", 500)

# ── dev entry --------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
