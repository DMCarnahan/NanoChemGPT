"""Vector store that embeds PDF chunks with OpenAI and stores them in a
FAISS index for similarity search.

* **Embeddings**: OpenAI `text-embedding-3-small` (1536‑D)
* **Index**: `faiss.IndexFlatIP` (cosine similarity after L2‑norm)
* **Persistence**: index + metadata JSON saved under `./data/`. Reloaded on
  import so uploads survive container restarts (good enough for dev).

Public API (used by *app.py*)
-----------------------------
add_to_store(text: str) -> None
    Splits into ~300‑token chunks, embeds, inserts into FAISS.

search(query: str, k: int = 4) -> str
    Embeds the query, performs similarity search, returns top‑k chunks
    concatenated (two blank lines separator).

stats() -> dict
    Quick corpus stats.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
from typing import List

import faiss  # type: ignore
import numpy as np
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import nltk
nltk.data.path.append("/opt/render/nltk_data")

from nltk.tokenize import sent_tokenize
# ── config ────────────────────────────────────────────────────────────────
load_dotenv()
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE  = DATA_DIR / "meta.json"  # list[str] chunks parallel to vectors

EMBED_MODEL = "text-embedding-3-small"
TOKENIZER   = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKS  = 300

_lock  = threading.Lock()  # Flask’s default worker model is threaded

# ── load / create index ───────────────────────────────────────────────────

def _new_index(dim: int = 1024) -> faiss.IndexFlatIP:
    idx = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(idx)

if INDEX_FILE.exists():
    index: faiss.IndexIDMap2 = faiss.read_index(str(INDEX_FILE))  # type: ignore[arg-type]
else:
    index = _new_index()

if META_FILE.exists():
    _meta: List[str] = json.loads(META_FILE.read_text())
else:
    _meta = []
assert len(_meta) == index.ntotal, "Index / metadata length mismatch"

# ── helpers ───────────────────────────────────────────────────────────────

from nltk.tokenize import sent_tokenize

def _chunk(text: str) -> list[str]:
    """
    Split *text* into ~CHUNK_TOKS-token passages that end on whole sentences.

    Uses NLTK's sent_tokenize, then greedily packs sentences until adding the
    next one would exceed CHUNK_TOKS tokens.
    """
    sentences = sent_tokenize(text)
    chunks, buf = [], []

    def buf_tokens() -> int:
        return len(TOKENIZER.encode(" ".join(buf)))

    for sent in sentences:
        if not sent.strip():
            continue
        buf.append(sent)
        if buf_tokens() >= CHUNK_TOKS:
            # If the single sentence itself is longer than CHUNK_TOKS,
            # keep it as its own chunk; else move it to new buffer
            if len(buf) == 1:
                chunks.append(" ".join(buf))
                buf = []
            else:
                # remove last sentence, flush, start new chunk with it
                last = buf.pop()
                chunks.append(" ".join(buf))
                buf = [last]

    if buf:
        chunks.append(" ".join(buf))

    return chunks
  
embedder = SentenceTransformer("intfloat/e5-large-v2")

def _embed_passages(texts: list[str]) -> np.ndarray:
    return embedder.encode(
        [f"passage: {t}" for t in texts],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

def _embed_query(text: str) -> np.ndarray:
    return embedder.encode(
        f"query: {text}",
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

def _persist() -> None:
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(_meta))

# ── public API ────────────────────────────────────────────────────────────

def add_to_store(doc: str) -> None:
    if not doc:
        return
    chunks = _chunk(doc)
    vecs = _embed_passages(chunks) 
    with _lock:
        start = len(_meta)
        ids   = np.arange(start, start + len(chunks)).astype("int64")
        index.add_with_ids(vecs, ids)
        _meta.extend(chunks)
        _persist()
        print(f"[vector_store] indexed {len(chunks)} chunks (total {len(_meta)})")

def search(query: str, k: int = 4) -> str:
    if index.ntotal == 0 or not query:
        return ""
    qvec = _embed_query(query)
    with _lock:
        scores, ids = index.search(qvec, min(k, index.ntotal))
    hits = [_meta[i] for i in ids[0] if i != -1]
    if hits:
        print(f"[vector_store] search → {len(hits)} hits (top score {scores[0][0]:.3f})")
    else:
        print("[vector_store] search → 0 hits")
    return "\n\n".join(hits)

def stats() -> dict:
    return {"chunks": len(_meta), "vectors": int(index.ntotal)}

# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Stats before:", stats())
    if not index.ntotal:
        add_to_store("This is a test doc about gold nanoparticles in ethylene glycol.")
    print(search("gold nanoparticles"))
