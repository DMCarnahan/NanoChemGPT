from __future__ import annotations
import os, time, threading, gzip, json, pathlib
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

# ---------------- Config ----------------
DATA_DIR = pathlib.Path(os.getenv("VECTORSTORE_DIR", "/tmp/index")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "index"; INDEX_DIR.mkdir(parents=True, exist_ok=True)

TTL_SEC        = int(os.getenv("UPLOAD_TTL_SEC", "1800"))   # 30 min
MODEL_NAME     = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BACKEND  = os.getenv("EMBED_BACKEND", "st")           # "st" | "openai"
EMB_BATCH      = int(os.getenv("EMBED_BATCH", "64"))
DEFER_EMBED    = os.getenv("DEFER_EMBED", "1") == "1"       # defer during preload by default

# ---------------- State ----------------
_model = None                  # sentence-transformers model OR None
_index: Optional[faiss.Index] = None
_meta: List[Dict[str, Any]] = []      # [{id, tag, ts, text}]
_dirty_index = False          
_lock = threading.Lock()

# ---------------- Model loading (lazy) ----------------
_ST_MODEL = None

def _load_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        # Lazy import to avoid torch import unless needed
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return _ST_MODEL

def _encode_st(texts: List[str]) -> np.ndarray:
    model = _load_st_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype="float32")

def _encode_openai(texts: List[str]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    out: List[List[float]] = []
    B = 64
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_OPENAI_MODEL, input=chunk)
        out.extend([e.embedding for e in resp.data])
    arr = np.array(out, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / norms).astype("float32")

def _encode(texts: List[str]) -> np.ndarray:
    try:
        if EMBED_BACKEND == "openai":
            return _encode_openai(texts)
        return _encode_st(texts)
    except Exception as e:
        print("[vector_store] embed failed:", e, "— set EMBED_BACKEND=openai to avoid local torch.")
        raise

# ---------------- Index IO ----------------
def _get_index(d: int) -> faiss.Index:
    global _index
    if _index is None:
        ipath = INDEX_DIR / "index.faiss"
        if ipath.exists():
            try:
                _index = faiss.read_index(str(ipath))
                _load_meta()
                return _index
            except Exception as e:
                print("[vector_store] read_index failed, rebuilding:", e)
        _index = faiss.IndexFlatIP(d)
    return _index

def _reset_index(d: int):
    global _index
    _index = faiss.IndexFlatIP(d)

def _persist():
    try:
        if _index is not None:
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

# ---------------- Chunking ----------------
def _chunk(text: str) -> List[str]:
    # Split into paragraph-ish chunks, hard cap
    parts = [para.strip()[:4000] for para in text.split("\n\n") if para.strip()]
    return parts or [text[:4000]]

# ---------------- Embedding build (lazy/full rebuild) ----------------
def _rebuild_index_locked():
    """(Re)embed all meta texts and rebuild FAISS. Caller must hold _lock."""
    global _dirty_index
    texts = [m["text"] for m in _meta]
    if not texts:
        _reset_index(1536)
        _persist()
        _dirty_index = False
        return

    embs = _encode(texts)
    d = embs.shape[1]
    _reset_index(d)
    _index.add(embs)
    _persist()
    _dirty_index = False
    print(f"[vector_store] rebuilt index with {len(_meta)} chunks; dim={d}")

def _ensure_index():
    with _lock:
        if _index is None:
            # try load existing; if not, build
            ipath = INDEX_DIR / "index.faiss"
            if ipath.exists():
                try:
                    _get_index(1536)  # dim ignored when reading
                except Exception:
                    pass
        # if index exists but out of sync or marked dirty 
        if _index is None or _dirty_index or (_index.ntotal != len(_meta)):
            _rebuild_index_locked()

# ---------------- Public API ----------------
def add_to_store(text: str, tag: str = "upload", defer_embed: Optional[bool] = None):
    """Add text to store. If defer_embed=True, we won't embed immediately."""
    if defer_embed is None:
        defer_embed = DEFER_EMBED

    chunks = _chunk(text)
    with _lock:
        cur = len(_meta)
        ts = int(time.time())
        for i, c in enumerate(chunks):
            _meta.append({"id": cur + i, "tag": tag, "ts": ts, "text": c})

        if defer_embed:
            # mark dirty; search() will rebuild lazily
            global _dirty_index
            _dirty_index = True
            _persist()
        else:
            # embed & add incrementally without full rebuild
            if _index is None or _index.ntotal != cur:
                # safest path: full rebuild to keep ids aligned
                _rebuild_index_locked()
            else:
                embs = _encode(chunks)
                _get_index(embs.shape[1]).add(embs)
                _persist()

        print(f"[vector_store] indexed {len(chunks)} chunks (total {len(_meta)}) tag={tag}")

def clear_uploads():
    """Remove transient upload:* chunks; mark index dirty so it rebuilds lazily."""
    with _lock:
        keep = [m for m in _meta if not str(m.get("tag","")).startswith("upload:")]
        _meta[:] = keep
        global _dirty_index
        _dirty_index = True
        _persist()
        print(f"[vector_store] cleared uploads; kept {len(_meta)} chunks")

def search(query: str, k: int = 8) -> str:
    _ensure_index()
    with _lock:
        if not _meta or _index is None or _index.ntotal == 0:
            return ""
        q = _encode([query])
        D, I = _index.search(q, min(k, _index.ntotal))
        lines = []
        for idx in I[0]:
            if 0 <= idx < len(_meta):
                m = _meta[idx]
                tag = m.get("tag", "ctx")
                # make source explicit
                lines.append(f"[SRC {tag}] {m['text']}")
        return "\n---\n".join(lines)


# ---------------- Expirer (uploads TTL) ----------------
def _expire_uploads():
    while True:
        time.sleep(60)
        now = int(time.time())
        changed = False
        with _lock:
            keep = []
            for m in _meta:
                tag = str(m.get("tag",""))
                if tag.startswith("upload:") and now - int(m.get("ts", now)) > TTL_SEC:
                    changed = True
                    continue
                keep.append(m)
            if changed:
                _meta[:] = keep
                global _dirty_index
                _dirty_index = True
                _persist()
                print("[vector_store] expired old uploads; marked index dirty")
threading.Thread(target=_expire_uploads, daemon=True).start()