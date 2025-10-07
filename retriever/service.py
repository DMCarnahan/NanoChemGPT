from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Lazy-import faiss to avoid import-time failures on environments where faiss
# is not installed or incompatible with the local NumPy ABI. Use HAS_FAISS and
# _faiss to refer to the module when available.
try:
    import faiss as _faiss

    HAS_FAISS = True
except Exception as _e:
    _faiss = None
    HAS_FAISS = False
    # don't print here; tests/CI will surface errors when FAISS functionality is used
try:
    from fastapi import Body, FastAPI
    from pydantic import BaseModel

    _HAS_FASTAPI = True
except Exception:
    # Provide minimal stubs so module can be imported in environments without fastapi
    _HAS_FASTAPI = False

    class Body:
        def __init__(self, default=None):
            self.default = default

    class _DummyApp:
        def __init__(self, *args, **kwargs):
            pass

        def on_event(self, *args, **kwargs):
            def _d(fn):
                return fn

            return _d

        def get(self, *args, **kwargs):
            def _d(fn):
                return fn

            return _d

        def post(self, *args, **kwargs):
            def _d(fn):
                return fn

            return _d

    FastAPI = _DummyApp
    from pydantic import BaseModel

from vector_store import embed  # returns List[List[float]]
from app_utils.constants import INDEX_DIR as CONST_INDEX_DIR

log = logging.getLogger("retriever")
log.setLevel(logging.INFO)

app = FastAPI(title="NanoChemGPT Retriever")

# ---- env & paths ----
# Prefer environment override, otherwise use a repo-local path to avoid
# attempting to write to root-level folders like '/data' (which CI often
# forbids). If creation fails due to permissions, fall back to a temp dir.
REPO_ROOT = Path(__file__).resolve().parents[1]
# Default index dir: prefer canonical constant but keep repo-local fallback
_default_index = REPO_ROOT / "vector_store"
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(CONST_INDEX_DIR or _default_index)))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(INDEX_DIR / "index.faiss")))
TEXTS_PATH = Path(os.getenv("KB_TEXTS_PATH", str(INDEX_DIR / "texts.jsonl")))
# default bundle inside the repo harvester output (safer than /data/out)
_default_bundle = REPO_ROOT / "harvester" / "out_auto" / "bundle.jsonl"
BUNDLE_PATH = Path(os.getenv("BUNDLE_PATH", str(_default_bundle)))

# Try creating the index/text dirs; if permission denied, fall back to a
# repo-local dir or a temporary directory to make the module importable in CI.
try:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_PATH.parent.mkdir(parents=True, exist_ok=True)
except PermissionError:
    # Try using repo-local fallback
    try:
        INDEX_DIR = _default_index
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        TEXTS_PATH = Path(INDEX_DIR) / "texts.jsonl"
        TEXTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Last resort: use an OS temp dir
        tmpd = tempfile.mkdtemp(prefix="nanochem_vectorstore_")
        logging.getLogger("retriever").warning(
            "Permission denied creating index dir; falling back to temp dir %s", tmpd
        )
        INDEX_DIR = Path(tmpd)
        FAISS_INDEX_PATH = Path(INDEX_DIR / "index.faiss")
        TEXTS_PATH = Path(INDEX_DIR / "texts.jsonl")

_state: Dict[str, Any] = {"index": None, "texts": []}


def _load_index() -> None:
    if FAISS_INDEX_PATH.exists():
        if not HAS_FAISS:
            # can't load index without faiss; leave state empty and log
            log.warning(
                "FAISS not available; skipping index load from %s", FAISS_INDEX_PATH
            )
            return
        idx = _faiss.read_index(str(FAISS_INDEX_PATH))
        texts = []
        if TEXTS_PATH.exists():
            with open(TEXTS_PATH, "r", encoding="utf-8") as f:
                texts = [json.loads(line) for line in f if line.strip()]
        _state["index"] = idx
        _state["texts"] = texts
        log.info(
            "Loaded FAISS index: ntotal=%s | texts=%s",
            getattr(idx, "ntotal", 0),
            len(texts),
        )
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

    if not HAS_FAISS:
        raise RuntimeError(
            "faiss is not available in this environment; cannot build index from bundle"
        )
    vecs = embed(texts)  # List[List[float]]
    X = np.asarray(vecs, dtype="float32")
    _faiss.normalize_L2(X)
    index = _faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _faiss.write_index(index, str(FAISS_INDEX_PATH))
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
    # Optional FAISS auto-build (explicit rebuild) if env set and bundle exists
    if os.getenv("AUTOBUILD_FAISS", "0").lower() in {"1", "true", "yes"}:
        try:
            if BUNDLE_PATH.exists() and BUNDLE_PATH.stat().st_size > 0:
                if _state.get("index") is None or int(getattr(_state["index"], "ntotal", 0) or 0) == 0:
                    log.info("[autobuild-faiss] rebuilding from bundle %s", BUNDLE_PATH)
                    _build_index_from_bundle(BUNDLE_PATH)
        except Exception as e:
            log.warning("[autobuild-faiss] failed: %s", e)


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
def reindex(bundle_path: Optional[str] = None, text_key: str = "text"):
    _build_index_from_bundle(
        Path(bundle_path) if bundle_path else BUNDLE_PATH, text_key=text_key
    )
    return {"ok": True, "ntotal": int(_state["index"].ntotal)}


# Accept missing body without 422 by making it Optional w/ default None
@app.post("/search")
def search(inp: Optional[SearchIn] = Body(default=None)):
    q = inp.q_text() if inp else ""
    if not q:
        return {"hits": [], "warning": "empty_query"}

    idx = _state["index"]
    if idx is None or int(getattr(idx, "ntotal", 0) or 0) <= 0:
        log.warning("[retriever] search fallback: no index loaded (q=%r)", q)
        return {"hits": [], "warning": "no_index"}

    # Clamp k to index size (and >=1)
    ntotal = int(getattr(idx, "ntotal", 0) or 0)
    kk = max(1, min((inp.k if inp else 8), ntotal))

    # Embed & search
    qv = np.asarray([embed([q])[0]], dtype="float32")
    if HAS_FAISS:
        _faiss.normalize_L2(qv)
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
