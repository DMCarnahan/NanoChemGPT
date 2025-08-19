
from __future__ import annotations
"""
kb.py — concrete implementations for kb_search() and kb_fetch() using:
  - FAISS vector index
  - JSONL doc store produced by the miner
  - Pluggable embedder (existing project → OpenAI → sentence-transformers)

Environment variables:
  FAISS_INDEX_PATH   : path to .faiss index file (default: data/index.faiss)
  FAISS_IDS_PATH     : path to ids.json (list[str] aligned to FAISS rows)
  DOCS_JSONL         : path to miner output JSONL (default: data/docs.jsonl)
  DOCS_IDX_PATH      : optional path for the sidecar id→offset index (default: DOCS_JSONL + ".idx.json")
  FAISS_METRIC       : "ip" (inner product / cosine) or "l2" (default: "ip")
  EMBED_BACKEND      : "project" | "openai" | "st"  (auto-detect if unset)
  OPENAI_EMB         : OpenAI embedding model name (default: text-embedding-3-small)
  EMBED_MODEL        : sentence-transformers model id (default: sentence-transformers/all-MiniLM-L6-v2)

Return format:
  kb_search(query, topk) -> [{"id": str, "sim": float}, ...]
  kb_fetch(ids) -> [doc_json, ...] where each doc_json includes:
    - "id": str
    - "meta": dict (may include year/source/url; we merge in if present at top-level)
    - the rest of your miner's normalized fields
"""
import os, json, io, numpy as np
from typing import List, Dict, Any, Iterable, Tuple

# ----------------- Config -----------------
INDEX_PATH = os.getenv("BUILTIN_DATA")
IDS_PATH   = os.getenv("FAISS_IDS_PATH", "data/ids.json")
DOCS_JSONL = os.getenv("DOCS_JSONL", "data/docs.jsonl")
DOCS_IDX_PATH = os.getenv("DOCS_IDX_PATH", DOCS_JSONL + ".idx.json")
FAISS_METRIC = os.getenv("FAISS_METRIC", "ip")  # "ip" or "l2"

# Embedding config
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "").strip().lower()  # "", "project", "openai", "st"
OPENAI_EMB = os.getenv("OPENAI_EMB", "text-embedding-3-small")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------- FAISS -----------------
try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss is required for kb_search(). Please add faiss-cpu to requirements.") from e

# Load FAISS index and row-aligned IDs at import-time
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
_index = faiss.read_index(INDEX_PATH)

if not os.path.exists(IDS_PATH):
    raise FileNotFoundError(f"IDs file not found at {IDS_PATH}")
with open(IDS_PATH, "r", encoding="utf-8") as f:
    _ids: List[str] = json.load(f)
# Optional sanity check
if hasattr(_index, "ntotal") and _index.ntotal != len(_ids):
    # allow but warn (no logger here to keep this lightweight)
    pass

def _normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / (n + 1e-12)

def _to_sim(dist: float) -> float:
    """Map FAISS distance to [0,1] similarity for the judge layer."""
    if FAISS_METRIC == "ip":
        # assume vectors were normalized; dot ∈ [0,1]
        return float(dist)
    # for L2, smaller distance = closer; map to bounded similarity
    return float(1.0 / (1.0 + max(dist, 0.0)))

# ----------------- Embedding -----------------
_model_cache = {}

def _try_project_embedder(q: str) -> np.ndarray | None:
    """Try to use the project's existing embedder if available."""
    candidates = [
        ("vector_store", "embed_query"),
        ("vector_store.v2", "embed_query"),
        ("app.vector_store", "embed_query"),
        ("app.vector_store.v2", "embed_query"),
    ]
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                v = fn(q)
                v = np.asarray(v, dtype="float32")
                return v
        except Exception:
            continue
    return None

def _embed_openai(q: str) -> np.ndarray:
    try:
        from openai import OpenAI  # openai>=1.0
    except Exception as e:
        raise RuntimeError("OpenAI client not available; set EMBED_BACKEND=st or install openai.") from e
    client = _model_cache.get("openai_client")
    if client is None:
        client = OpenAI()
        _model_cache["openai_client"] = client
    resp = client.embeddings.create(model=OPENAI_EMB, input=q)
    vec = np.array(resp.data[0].embedding, dtype="float32")
    return vec

def _embed_st(q: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not available; set EMBED_BACKEND=openai or install it.") from e
    model = _model_cache.get("st_model")
    if model is None:
        model = SentenceTransformer(EMBED_MODEL)
        _model_cache["st_model"] = model
    vec = model.encode([q])[0].astype("float32")
    return vec

def embed_query(q: str) -> np.ndarray:
    """Return a float32 vector (d,) for the query string."""
    # 1) Prefer project's embedder if EMBED_BACKEND empty or "project"
    if EMBED_BACKEND in ("", "project"):
        v = _try_project_embedder(q)
        if v is not None:
            return v
        if EMBED_BACKEND == "project":
            try:
                return _embed_openai(q)
            except Exception:
                return _embed_st(q)
    # 2) OpenAI
    if EMBED_BACKEND == "openai":
        return _embed_openai(q)
    # 3) sentence-transformers
    if EMBED_BACKEND == "st":
        return _embed_st(q)
    # 4) Auto: try OpenAI then ST
    try:
        return _embed_openai(q)
    except Exception:
        return _embed_st(q)

# ----------------- JSONL Doc Store -----------------
class JSONLDocStore:
    """
    Random-access JSONL store with an id→byte-offset sidecar index.

    JSONL lines may have either format A or B:
      A) {"id": "...", "json": {...normalized...}, "meta": {...}}
      B) {"id": "...", ...normalized fields..., "meta": {...}, "source": "...", "year": 2024, "url": "..."}
    We return a dict including at least {"id": str, "meta": dict, ...normalized fields...}.
    """
    def __init__(self, jsonl_path: str, idx_path: str | None = None):
        self.jsonl_path = jsonl_path
        self.idx_path = idx_path or (jsonl_path + ".idx.json")
        self._index: Dict[str, Tuple[int,int]] = {}  # id -> (start_offset, length)
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Docs JSONL not found at {self.jsonl_path}")
        self._load_or_build_index()

    def _load_or_build_index(self):
        if os.path.exists(self.idx_path):
            try:
                with open(self.idx_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
                    self._index = {k: tuple(v) for k,v in self._index.items()}
                    return
            except Exception:
                pass
        idx: Dict[str, Tuple[int,int]] = {}
        with open(self.jsonl_path, "rb") as f:
            pos = 0
            for line in f:
                if not line.strip():
                    pos += len(line); continue
                try:
                    o = json.loads(line.decode("utf-8"))
                except Exception:
                    pos += len(line); continue
                _id = o.get("id")
                if isinstance(_id, str):
                    idx[_id] = (pos, len(line))
                pos += len(line)
        with open(self.idx_path, "w", encoding="utf-8") as f:
            json.dump(idx, f)
        self._index = idx

    def fetch_many(self, ids: Iterable[str]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "rb") as f:
            for id_ in ids:
                off_len = self._index.get(id_)
                if not off_len:
                    continue
                start, length = off_len
                f.seek(start)
                raw = f.read(length)
                try:
                    o = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                doc = self._normalize_line_object(o)
                if doc:
                    docs.append(doc)
        return docs

    @staticmethod
    def _normalize_line_object(o: Dict[str, Any]) -> Dict[str, Any] | None:
        _id = o.get("id")
        if not isinstance(_id, str):
            return None
        meta = o.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        if "json" in o and isinstance(o["json"], dict):
            jdoc = dict(o["json"])
        else:
            jdoc = dict(o)
            jdoc.pop("json", None)
            for k in ("source", "year", "url"):
                if k in jdoc:
                    meta.setdefault(k, jdoc.pop(k))
        jdoc["id"] = _id
        jdoc["meta"] = meta
        return jdoc

_docstore = JSONLDocStore(DOCS_JSONL, DOCS_IDX_PATH)

def kb_search(query: str, topk: int = 8) -> List[Dict[str, Any]]:
    v = embed_query(query).astype("float32")
    if FAISS_METRIC == "ip":
        v = _normalize(v)
    D, I = _index.search(v[None,:], topk)
    hits: List[Dict[str, Any]] = []
    for dist, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1 or idx >= len(_ids):
            continue
        hits.append({"id": _ids[idx], "sim": _to_sim(dist)})
    return hits

def kb_fetch(ids: Iterable[str]) -> List[Dict[str, Any]]:
    return _docstore.fetch_many(ids)
