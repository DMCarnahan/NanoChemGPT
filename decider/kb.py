from __future__ import annotations
import os, json, logging
from pathlib import Path

logging.getLogger(__name__).setLevel(logging.INFO)

_raw = os.getenv("RETRIEVER_URL", "").strip()
port = os.getenv("PORT", "8080")  # fallback for local dev
RETRIEVER_URL = _raw.replace("{PORT}", port) if "{PORT}" in _raw else _raw
REMOTE_TIMEOUT = float(os.getenv("KB_REMOTE_TIMEOUT", "3.0"))

# Resolve repo root -> data/vector_store by default
APP_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(APP_ROOT / "data" / "vector_store")))
INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(INDEX_DIR / "index.faiss")))
TEXTS_PATH = Path(os.getenv("KB_TEXTS_PATH", str(INDEX_DIR / "texts.jsonl")))

_index = None
_texts = None
_loaded = False

def _lazy_load() -> None:
    """
    Load local FAISS index and texts only once and only if we're not using a remote retriever.
    Never raise on failure — just disable KB gracefully.
    """
    global _index, _texts, _loaded
    if _loaded or RETRIEVER_URL:
        _loaded = True
        return

    try:
        import faiss  # type: ignore
        if INDEX_PATH.exists():
            _index = faiss.read_index(str(INDEX_PATH))
            logging.info("KB: loaded FAISS index from %s (ntotal=%s)", INDEX_PATH, getattr(_index, "ntotal", "?"))
            if TEXTS_PATH.exists():
                with open(TEXTS_PATH, "r", encoding="utf-8") as f:
                    _texts = [json.loads(line) for line in f if line.strip()]
            else:
                _texts = None
                logging.warning("KB: %s not found; results will lack metadata.", TEXTS_PATH)
        else:
            logging.warning("KB: FAISS index not found at %s. Local KB disabled.", INDEX_PATH)
        _loaded = True
    except Exception:
        logging.exception("KB: failed to load local FAISS index; disabling KB.")
        _index = None
        _texts = None
        _loaded = True

def kb_available() -> bool:
    """Return True if either remote retriever is configured or local index is present."""
    if RETRIEVER_URL:
        return True
    _lazy_load()
    return _index is not None

def kb_search(query: str, k: int = 8) -> list[dict]:
    """
    Search KB.

    If RETRIEVER_URL is set, try POST {RETRIEVER_URL}/search first
    (expects {"hits":[...]}) and fall back to local FAISS on error or empty result.
    Otherwise, embed locally via vector_store.v2 and query FAISS.

    Returns: list of dicts with at least {"i": idx, "score": float, ...meta}
    """
    if not query or not query.strip():
        return []

    # ---- Remote first (co-hosted retriever or separate service) ----
    if RETRIEVER_URL:
        try:
            import requests  # type: ignore
            resp = requests.post(
                f"{RETRIEVER_URL.rstrip('/')}/search",
                json={"q": query, "k": int(k)},
                timeout=REMOTE_TIMEOUT,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits") or []
            if hits:
                return hits
            # If remote is healthy but has no hits (e.g., empty index), try local fallback
        except Exception:
            logging.exception("KB: remote retriever failed; attempting local fallback.")

    # ---- Local fallback (FAISS) ----
    _lazy_load()
    if _index is None or int(getattr(_index, "ntotal", 0) or 0) <= 0:
        return []

    from vector_store import embed  # uses EMBED_BACKEND/EMBED_MODEL/OPENAI_EMB envs
    import numpy as np
    import faiss  # type: ignore

    vec = embed([query])[0]
    xq = np.asarray([vec], dtype="float32")

    try:
        faiss.normalize_L2(xq)
    except Exception:
        pass

    ntotal = int(getattr(_index, "ntotal", 0) or 0)
    kk = min(int(k), ntotal) if ntotal > 0 else 0
    if kk <= 0:
        return []

    D, I = _index.search(xq, kk)

    hits: list[dict] = []
    idxs, scores = I[0].tolist(), [float(s) for s in D[0].tolist()]
    for idx, score in zip(idxs, scores):
        if idx < 0:
            continue
        meta = _texts[idx] if (_texts and 0 <= idx < len(_texts)) else {}
        hits.append({"i": idx, "score": score, **meta})
    return hits

def kb_fetch(idx: int) -> dict:
    """
    Fetch metadata/payload for a given local index row.
    In remote mode you should not call this; remote should return full payloads in kb_search.
    """
    _lazy_load()
    if _texts and 0 <= idx < len(_texts):
        return _texts[idx]
    return {}
