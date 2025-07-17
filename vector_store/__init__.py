"""Embeds PDF chunks with **all‑MiniLM‑L6‑v2** (384‑D) and stores them in
FAISS. 
"""
from __future__ import annotations

import json, pathlib, threading
from typing import List

import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken

DATA_DIR   = pathlib.Path("data"); DATA_DIR.mkdir(exist_ok=True)
INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE  = DATA_DIR / "meta.json"

TOKENIZER  = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKS = 300
DIM        = 384  # MiniLM

# lazy model load -----------------------------------------------------------
_embedder = None  # type: SentenceTransformer | None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("[vector_store] loading all-MiniLM-L6-v2 …")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

# helpers -------------------------------------------------------------------

def _json_to_lines(obj, prefix=""):
    """Yield 'key.path: value' lines from any nested mapping / list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _json_to_lines(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _json_to_lines(v, f"{prefix}{i}.")
    else:                                  # leaf → value
        yield f"{prefix[:-1]}: {obj}"

def _chunk(text: str) -> List[str]:
    words, out, buf = text.split(), [], []
    for w in words:
        buf.append(w)
        if len(TOKENIZER.encode(" ".join(buf))) >= CHUNK_TOKS:
            out.append(" ".join(buf)); buf = []
    if buf: out.append(" ".join(buf))
    return out


def _embed_passages(txts: list[str]) -> np.ndarray:
    emb = _get_embedder()
    vecs = emb.encode([f"passage: {t}" for t in txts], normalize_embeddings=True,
                      convert_to_numpy=True).astype("float32")
    return vecs


def _embed_query(q: str) -> np.ndarray:
    emb = _get_embedder()
    return emb.encode(f"query: {q}", normalize_embeddings=True,
                      convert_to_numpy=True).astype("float32")

import json as _json

def add_json_to_store(json_bytes: bytes):
    """Convert JSON → lines → one doc string, then embed."""
    try:
        data = _json.loads(json_bytes)
    except ValueError as e:
        print("JSON parse error:", e)
        return
    doc_text = "\n".join(_json_to_lines(data))
    add_to_store(doc_text)

# index init ----------------------------------------------------------------

def _new_index() -> faiss.IndexIDMap2:
    return faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))

def _load() -> tuple[faiss.IndexIDMap2, list[str]]:
    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))  # type: ignore[arg-type]
        meta  = json.loads(META_FILE.read_text())
    else:
        index, meta = _new_index(), []
    return index, meta

_index, _meta = _load()
_lock = threading.Lock()

# persistence ----------------------------------------------------------------

def _persist():
    faiss.write_index(_index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(_meta))

# builtin data -----------------------------------------------------------------------

import pathlib, json, csv, gzip
from PyPDF2 import PdfReader

BUILTIN_DIR = pathlib.Path(__file__).parent / "builtin_data"

def _load_builtin_once():
    """Embed every file under builtin_data/ on first launch."""
    if _index.ntotal:       # index already has vectors → skip
        return
    for path in BUILTIN_DIR.rglob("*"):
        if path.suffix == ".json":
            add_json_bytes(path.read_bytes())
        elif path.suffix == ".csv":
            add_csv_bytes(path.read_bytes())
        elif path.suffix == ".gz" and path.name.endswith(".json.gz"):
            add_json_bytes(gzip.open(path, "rb").read())
        elif path.suffix == ".pdf":
            text = "\n".join(
                (PdfReader(str(path)).pages[i].extract_text() or "")
                for i in range(len(PdfReader(str(path)).pages))
            )
            add_to_store(text)
        # add more formats as you like
    print("[vector_store] builtin data embedded")

# public API -----------------------------------------------------------------

def add_to_store(doc: str):
    if not doc: return
    chunks = _chunk(doc)
    vecs   = _embed_passages(chunks)
    with _lock:
        ids = np.arange(len(_meta), len(_meta)+len(chunks)).astype("int64")
        _index.add_with_ids(vecs, ids)
        _meta.extend(chunks)
        _persist()
        print(f"[vector_store] indexed {len(chunks)} chunks (total {len(_meta)})")


def search(query: str, k: int = 4) -> str:
    if _index.ntotal == 0 or not query:
        return ""
    qvec = _embed_query(query)
    with _lock:
        scores, ids = _index.search(qvec, min(k, _index.ntotal))
    hits = [_meta[i] for i in ids[0] if i != -1]
    return "\n\n".join(hits)


def stats():
    return {"chunks": len(_meta), "vectors": int(_index.ntotal)}

# cli -----------------------------------------------------------------------
if __name__ == "__main__":
    print(stats())
# builtin load -----------------------------------------------------------------------
_load_builtin_once()
