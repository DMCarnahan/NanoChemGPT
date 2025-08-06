from __future__ import annotations
import os
import time
import threading
import gzip
import json
import pathlib
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

DATA_DIR = pathlib.Path(os.getenv("VECTORSTORE_DIR", "/tmp/index"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
TTL_SEC = int(os.getenv("UPLOAD_TTL_SEC", "1800"))  # 30 min
MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/e5-large-v2")

_model = None
_index = None
_meta: List[Dict[str, Any]] = []
_lock = threading.Lock()

def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def _encode(texts: List[str]) -> np.ndarray:
    model = _load_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype='float32')

def _get_index(d: int):
    global _index
    if _index is None:
        _index = faiss.IndexFlatIP(d)
        ipath = INDEX_DIR / "index.faiss"
        if ipath.exists():
            try:
                _index = faiss.read_index(str(ipath))
                _load_meta()
            except Exception:
                pass
    return _index

def _persist():
    try:
        faiss.write_index(_index, str(INDEX_DIR / "index.faiss"))
        with gzip.open(INDEX_DIR / "meta.json.gz", "wt", encoding="utf-8") as f:
            json.dump(_meta, f)
    except Exception as e:
        print("[vector_store] persist error:", e)

def _load_meta():
    global _meta
    p = INDEX_DIR / "meta.json.gz"
    if p.exists():
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                _meta = json.load(f)
        except Exception:
            _meta = []

def _chunk(text: str) -> List[str]:
    """Split text into ~paragraph chunks, max 4000 chars each."""
    parts = [para.strip()[:4000] for para in text.split("\n\n") if para.strip()]
    return parts or [text[:4000]]

def add_to_store(text: str, tag: str = "upload"):
    with _lock:
        chunks = _chunk(text)
        embs = _encode(chunks)
        index = _get_index(embs.shape[1])
        cur = len(_meta)
        index.add(embs)
        ts = int(time.time())
        for i, c in enumerate(chunks):
            _meta.append({"id": cur + i, "tag": tag, "ts": ts, "text": c})
        _persist()
        print(f"[vector_store] indexed {len(chunks)} chunks (total {len(_meta)})")

def clear_uploads():
    with _lock:
        keep = [m for m in _meta if not str(m.get("tag", "")).startswith("upload:")]
        texts = [m["text"] for m in keep]
        if not texts:
            d = _get_index(1).d if _index else 1536
            _reset_index(d)
            _meta[:] = []
            _persist()
            return
        embs = _encode(texts)
        d = embs.shape[1]
        _reset_index(d)
        _index.add(embs)
        _meta[:] = keep
        _persist()

def _reset_index(d: int):
    global _index
    _index = faiss.IndexFlatIP(d)

def _expire_uploads():
    while True:
        time.sleep(60)
        now = int(time.time())
        changed = False
        with _lock:
            keep = []
            for m in _meta:
                tag = str(m.get("tag", ""))
                if tag.startswith("upload:") and now - int(m.get("ts", now)) > TTL_SEC:
                    changed = True
                    continue
                keep.append(m)
            if changed:
                texts = [m["text"] for m in keep]
                if texts:
                    embs = _encode(texts)
                    d = embs.shape[1]
                    _reset_index(d)
                    _index.add(embs)
                else:
                    _reset_index(1536)
                _meta[:] = keep
                _persist()

threading.Thread(target=_expire_uploads, daemon=True).start()

def search(query: str, k: int = 4) -> str:
    with _lock:
        if not _meta or _index is None or _index.ntotal == 0:
            return ""
        q = _encode([query])
        D, I = _index.search(q, min(k, _index.ntotal))
        lines = []
        for idx in I[0]:
            if 0 <= idx < len(_meta):
                lines.append(_meta[idx]["text"])
        return "\n---\n".join(lines)
    
    from pymongo import MongoClient
import os

def preload_builtin_from_mongo():
    uri = os.getenv("MONGO_URL")
    if not uri: 
        print("[preload] no MONGO_URL; skip"); return
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client.get_default_database()
    col = db[os.getenv("BUILTIN_COLLECTION", "builtin_docs")]

    count = 0
    for d in col.find({}, {"text":1, "_id":0}):
        txt = (d.get("text") or "").strip()
        if txt:
            add_to_store(txt, tag="builtin:mongo")
            count += 1
    print(f"[preload] indexed {count} builtin docs from Mongo")
