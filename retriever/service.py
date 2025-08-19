from __future__ import annotations
import os, json, logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss  # pip install faiss-cpu
from fastapi import FastAPI, Body
from pydantic import BaseModel

from vector_store import embed  # returns List[List[float]]

log = logging.getLogger("retriever")
log.setLevel(logging.INFO)

app = FastAPI(title="NanoChemGPT Retriever")

# ---- env & paths ----
INDEX_DIR = Path(os.getenv("INDEX_DIR", "/data/vector_store"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(INDEX_DIR / "index.faiss")))
TEXTS_PATH = Path(os.getenv("KB_TEXTS_PATH", str(INDEX_DIR / "texts.jsonl")))
BUNDLE_PATH = Path(os.getenv("BUNDLE_PATH", "/data/out/bundle.jsonl"))

INDEX_DIR.mkdir(parents=True, exist_ok=True)
TEXTS_PATH.parent.mkdir(parents=True, exist_ok=True)

_state: Dict[str, Any] = {"index": None, "texts": []}

def _load_index() -> None:
    if FAISS_INDEX_PATH.exists():
        idx = faiss.read_index(str(FAISS_INDEX_PATH))
        texts = []
        if TEXTS_PATH.exists():
            with open(TEXTS_PATH, "r", encoding="utf-8") as f:
                texts = [json.loads(line) for line in f if line.strip()]
        _state["index"] = idx
        _state["texts"] = texts
        log.info("Loaded FAISS index: ntotal=%s | texts=%s", getattr(idx, "ntotal", 0), len(texts))
    else:
        log.warning("FAISS index not found at %s", FAISS_INDEX_PATH)

def _build_index_from_bundle(bundle_path: Path) -> None:
    if not bundle_path.exists() or bundle_path.stat().st_size == 0:
        raise FileNotFoundError(f"bundle not found or empty: {bundle_path}")
    docs: List[Dict[str, Any]] = []
    with open(bundle_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    # Accept either "text" or "content"
    texts = [(d.get("text") or d.get("content") or "").strip() for d in docs]
    texts = [t for t in texts if t]  # drop empties
    if not texts:
        raise ValueError("no text found in bundle; expected 'text' or 'content' field")

    vecs = embed(texts)  # List[List[float]]
    X = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    _state["index"] = index
    _state["texts"] = docs
    log.info("Built index from bundle: ntotal=%s", index.ntotal)

@app.on_event("startup")
def startup():
    _load_index()
    # Bootstrap from bundle if index is missing and bundle exists
    if _state["index"] is None and BUNDLE_PATH.exists():
        try:
            _build_index_from_bundle(BUNDLE_PATH)
        except Exception as e:
            log.error("Bootstrap build failed: %s", e)

class SearchIn(BaseModel):
    # Accept multiple aliases; all optional so missing body won't 422
    q: Optional[str] = None
    query: Optional[str] = None
    question: Optional[str] = None
    k: int = 8

    def q_text(self) -> str:
        return (self.q or self.query or self.question or "").strip()

@app.get("/health")
def health():
    ntotal = int(getattr(_state["index"], "ntotal", 0) or 0)
    return {"ok": True, "ntotal": ntotal, "texts": len(_state["texts"])}

@app.post("/reindex")
def reindex(bundle_path: Optional[str] = None):
    p = Path(bundle_path) if bundle_path else BUNDLE_PATH
    _build_index_from_bundle(p)
    return {"ok": True, "ntotal": int(_state['index'].ntotal)}

# Accept missing body without 422 by making it Optional w/ default None
@app.post("/search")
def search(inp: Optional[SearchIn] = Body(default=None)):
    q = inp.q_text() if inp else ""
    if not q:
        return {"hits": []}

    idx = _state["index"]
    if idx is None or int(getattr(idx, "ntotal", 0) or 0) <= 0:
        return {"hits": []}

    # Clamp k to index size (and >=1)
    ntotal = int(getattr(idx, "ntotal", 0) or 0)
    kk = max(1, min((inp.k if inp else 8), ntotal))

    # Embed & search
    qv = np.asarray([embed([q])[0]], dtype="float32")
    faiss.normalize_L2(qv)
    D, I = idx.search(qv, kk)

    hits: List[Dict[str, Any]] = []
    for i, s in zip(I[0].tolist(), D[0].tolist()):
        if i < 0:
            continue
        meta = _state["texts"][i] if 0 <= i < len(_state["texts"]) else {}
        hits.append({"i": i, "score": float(s), **meta})
    return {"hits": hits}

@app.get("/search")
def search_get(q: str, k: int = 8):
    return search(SearchIn(q=q, k=k))
