from __future__ import annotations

import io
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, abort
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import tiktoken

import vector_store as vs
from backend.parser import convert_to_json, ParserError

# ── basic setup ────────────────────────────────────────────────────────────
load_dotenv()
app = Flask(__name__, template_folder="templates", static_folder="static")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ENC         = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL = "text-embedding-3-small"
MAX_TOKENS  = 256

# ── utilities ──────────────────────────────────────────────────────────────

def _chunk(text: str, max_toks: int = 300) -> List[str]:
    words, buf, out = text.split(), [], []
    for w in words:
        buf.append(w)
        if len(ENC.encode(" ".join(buf))) >= max_toks:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out

from sentence_transformers import SentenceTransformer
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        print("[vector_store] loading all-MiniLM-L6-v2 …")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def _embed(texts: list[str]) -> np.ndarray:
    emb = _get_embedder()
    vecs = emb.encode(
        [f"query: {t}" if i == 0 else f"passage: {t}" for i, t in enumerate(texts)],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")
    return vecs

def _extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# ── routes ────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")
    if not file or file.filename == "":
        abort(400, "No PDF supplied.")
    if file.mimetype != "application/pdf":
        abort(400, "Only PDFs accepted.")

    text = _extract_text(file.read())
    try:
        vs.add_to_store(text)
    except Exception as err:  # noqa: BLE001
        print("[vector_store] add_to_store failed:", err)
    return jsonify({"status": "ok", "filename": file.filename})


@app.route("/ask", methods=["POST"])
def ask():
    """LLM Q&A with robust error handling so `answer` is always defined."""
    q = request.form.get("question", "").strip()
    if not q:
        abort(400, "No question.")

    try:
        context = vs.search(q, k=4)
    except Exception as err:  # noqa: BLE001
        print("[vector_store] search failed:", err)
        context = ""

    prompt = (
        "You are ChatAuNP, an AI assistant that designs gold nanomaterial syntheses. "
        "Use the provided context unless general chemistry knowledge is required. "
        "Provide concrete numerical parameters on the same volume scale as the paper. "
        "Response format (replace []):\n\n"
        "1. **Materials**: \n[]\n"
        "2. **Procedure**\n[]\n"
        "3. **Characterization**:\n[]\n\n"
        f"Context:\n{context}\n\nUser question: {q}"
    )

    resp = client.chat.completions.create(…)
    raw = resp.choices[0].message.content
    
    # split answer vs rationale
    if "```reason" in raw:
        answer, rationale_block = raw.split("```reason", 1)
        rationale = rationale_block.split("```", 1)[0].strip()
    else:
        answer, rationale = raw, ""
    
    return jsonify({"answer": answer.strip(),
                    "rationale": rationale})


@app.route("/parse", methods=["POST"])
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


@app.route("/ping")
def ping():
    return jsonify({"status": "alive"})

import io
from flask import send_file
from datetime import datetime

@app.post("/save_txt")
def save_txt():
    data = request.get_json(silent=True) or {}
    answer   = data.get("answer", "").strip()
    question = data.get("question", "").strip()
    if not answer:
        abort(400, "answer is empty")

    full_text = f"Q: {question}\n\nA:\n{answer}\n"
    buf = io.BytesIO(full_text.encode("utf-8"))
    buf.seek(0)

    filename = f"chatau_{datetime.utcnow():%Y%m%d_%H%M%S}.txt"
    return send_file(
        buf,
        mimetype="text/plain",
        as_attachment=True,
        download_name=filename,
    )

# ── unified JSON error handler ────────────────────────────────────────────
@app.errorhandler(400)
@app.errorhandler(422)
@app.errorhandler(502)
@app.errorhandler(500)
def handle_err(e):  # noqa: ANN001
    return jsonify(error=str(e)), getattr(e, "code", 500)

# ── local dev helper ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
