from __future__ import annotations

import os, json, math, pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

# ------------------------- Index path helpers -------------------------

def _index_dir() -> Path:
    """Resolve the index dir (env first, then default to <this>/index)."""
    env = os.getenv("RETRIEVER_INDEX_DIR")
    if env:
        return Path(env).resolve()
    return (Path(__file__).parent / "index").resolve()


# ---------------------------- TF‑IDF loader ---------------------------

@lru_cache(maxsize=1)
def _load_tfidf() -> Dict[str, Any]:
    """
    Load TF-IDF index from RETRIEVER_INDEX_DIR, accepting multiple layouts:
      - New:   tfidf.pkl            (joblib dump of {"matrix","vectorizer", ...})
      - Legacy:tfidf.npz+vectorizer.joblib
      - Very legacy: tfidf.npz+vocab.json  (bare vectorizer; idf_ may be missing)

    Returns a dict with at least:
      {"kind":"matrix","matrix":X,"vectorizer":vec,"texts":[...],"metas":[...]}
    and legacy aliases:
      {"vec": vectorizer, "nn": matrix}
    """
    idx = _index_dir()
    pkl  = idx / "tfidf.pkl"
    npz  = idx / "tfidf.npz"
    vecj = idx / "vectorizer.joblib"
    vocab = idx / "vocab.json"

    print(f"[retriever] Using INDEX_DIR={idx}")

    def _ensure_texts_metas(bundle: dict) -> dict:
        X = bundle.get("matrix")
        n = int(getattr(X, "shape", (0, 0))[0]) if X is not None else 0

        texts = bundle.get("texts")
        metas = bundle.get("metas")
        rows  = bundle.get("rows")  # legacy sidecar list of dicts

        # If rows provided, use them as defaults
        if rows and not texts:
            try:
                texts = [r.get("text", "") for r in rows if isinstance(r, dict)]
            except Exception:
                pass
        if rows and not metas:
            metas = rows

        def _as_list(x, n, fill):
            """Coerce x -> list length n. If dict with numeric keys, scatter by index."""
            if x is None:
                return [fill() for _ in range(n)]
            # tuple -> list
            if isinstance(x, tuple):
                x = list(x)
            # dict handling
            if isinstance(x, dict):
                # If dict keys look like indices, scatter into an array
                if all(str(k).isdigit() for k in x.keys()):
                    arr = [fill() for _ in range(n)]
                    for k, v in x.items():
                        try:
                            i = int(k)
                            if 0 <= i < n:
                                arr[i] = v
                        except Exception:
                            pass
                    x = arr
                else:
                    # single metadata dict replicated
                    x = [x] * max(1, n)
            # scalar / other → replicate
            if not isinstance(x, list):
                x = [x] * max(1, n)
            # pad/trim to length n
            if len(x) < n:
                x = x + [fill() for _ in range(n - len(x))]
            elif len(x) > n:
                x = x[:n]
            return x

        texts = _as_list(texts, n, lambda: "")
        # ensure all texts are strings
        texts = [t if isinstance(t, str) else str(t) for t in texts]

        metas = _as_list(metas, n, lambda: {})
        # ensure metas are dicts
        metas = [m if isinstance(m, dict) else {"meta": m} for m in metas]

        bundle["texts"] = texts
        bundle["metas"] = metas
        # legacy aliases so older code using ["vec"],["nn"] keeps working
        bundle["vec"] = bundle.get("vectorizer")
        bundle["nn"]  = bundle.get("matrix")
        return bundle


    # ---- Preferred format: single pickle ----
    if pkl.exists():
        obj = joblib.load(pkl)
        if isinstance(obj, dict):
            X = _coalesce(obj.get("matrix"), obj.get("X"), obj.get("tfidf"))
            vectorizer = _coalesce(obj.get("vectorizer"), obj.get("vec"))
            if X is None or vectorizer is None:
                raise RuntimeError(f"Malformed {pkl}: expected 'matrix' and 'vectorizer'")            bundle = {"kind": "matrix", "matrix": X, "vectorizer": vectorizer}
            for k in ("texts","metas","rows","license","titles"):
                if k in obj:
                    bundle[k] = obj[k]
            return _ensure_texts_metas(bundle)
        else:
            raise RuntimeError(f"Unsupported object in {pkl}: {type(obj)}")

    # ---- Legacy format: npz + vectorizer.joblib ----
    if npz.exists():
        # Try SciPy sparse loader first
        try:
            from scipy.sparse import load_npz  # type: ignore
            X = load_npz(npz)
        except Exception as e:
           
            try:
                f = np.load(npz)
                key = "arr_0" if "arr_0" in f.files else next(iter(f.files))
                X = f[key]
            except Exception as ee:
                raise RuntimeError(
                    f"Found {npz} but could not load it as sparse or dense: {e} / {ee}. "
                    f"Install SciPy or convert to tfidf.pkl."
                )
        if vecj.exists():
            vectorizer = joblib.load(vecj)
            bundle = {"kind": "matrix", "matrix": X, "vectorizer": vectorizer}
     
            for sidecar in ("rows.pkl","rows.jsonl","texts.jsonl","meta.json"):
                p = idx / sidecar
                if not p.exists():
                    continue
                try:
                    if p.suffix == ".pkl":
                        rows = joblib.load(p)
                        if isinstance(rows, list):
                            bundle["rows"] = rows
                    elif p.suffix == ".json":
                        bundle["metas"] = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    else:
                        rows = []
                        with p.open("r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                try:
                                    rows.append(json.loads(line))
                                except Exception:
                                    pass
                        if rows:
                            bundle["rows"] = rows
                except Exception:
                    pass
            return _ensure_texts_metas(bundle)

        # ---- Very legacy: npz + vocab.json ----
        if vocab.exists():
            from sklearn.feature_extraction.text import TfidfVectorizer
            vocab_list = json.loads(vocab.read_text(encoding="utf-8", errors="ignore"))
            vocab_map = {t: i for i, t in enumerate(vocab_list)}
            vectorizer = TfidfVectorizer(vocabulary=vocab_map)
            bundle = {"kind":"matrix", "matrix": X, "vectorizer": vectorizer}
            return _ensure_texts_metas(bundle)

    raise RuntimeError(
        f"No TF-IDF index found in {_index_dir()}. Expected tfidf.pkl OR tfidf.npz(+vectorizer.joblib). "
        f"Set RETRIEVER_INDEX_DIR appropriately."
    )


# ----------------------------- Compatibility -----------------------------

class Embedder:
    """
    Minimal compatibility shim so legacy imports `from retriever.retriever import Embedder` work.
    Supports two backends: 'openai' and 'sentence-transformers'.
    """
    def __init__(self, backend: str | None = None, model: str | None = None):
        cfg = _load_embed()
        self.backend = backend or cfg.get("backend") or "sentence-transformers"
        self.model = model or cfg.get("model") or "sentence-transformers/all-MiniLM-L6-v2"
        self._st = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "openai":
            from openai import OpenAI  # requires OPENAI_API_KEY
            client = OpenAI()
            resp = client.embeddings.create(model=self.model, input=texts)
            arr = np.array([d.embedding for d in resp.data], dtype="float32")
            n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            return (arr / n).astype("float32")
        else:
            from sentence_transformers import SentenceTransformer
            if self._st is None:
                self._st = SentenceTransformer(self.model)
            vecs = self._st.encode(texts, normalize_embeddings=True)
            return vecs.astype("float32")

    def embed(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def _load_embed() -> Dict[str, Any]:
    """Optional embed config: index/embed.json or env flags."""
    cfg = {}
    idx = _index_dir()
    j = idx / "embed.json"
    if j.exists():
        try:
            cfg.update(json.loads(j.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            pass
    # env overrides
    if os.getenv("EMBED_BACKEND"):
        cfg["backend"] = os.getenv("EMBED_BACKEND")
    if os.getenv("EMBED_MODEL"):
        cfg["model"] = os.getenv("EMBED_MODEL")
    return cfg


# ------------------------------- Search API -------------------------------

def _get_vec_nn(tf: dict):
    vec = tf.get("vec")
    if vec is None:
        vec = tf.get("vectorizer")

    nn = tf.get("nn")
    if nn is None:
        nn = tf.get("matrix")
    if nn is None:
        nn = tf.get("X")
    if nn is None:
        nn = tf.get("tfidf")

    if vec is None or nn is None:
        raise RuntimeError(f"Bad TF-IDF bundle keys: {list(tf.keys())}")
    return vec, nn

def _cosine_sim(query_vec, matrix):
    """Compute cosine similarity scores between query_vec (1 x d) and matrix (N x d)."""
    # If sparse:
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(matrix):
            # Normalize if needed
            q = query_vec
            if sp.issparse(q):
                q_norm = np.sqrt(q.multiply(q).sum())
                if q_norm > 0:
                    q = q / q_norm
            # Assume matrix rows are L2-normalized if indexer did so; if not, normalize on the fly
            scores = (matrix @ q.T).toarray().ravel()
            m_norm = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
            nz = (m_norm > 0)
            if not np.allclose(m_norm[nz], 1.0, atol=1e-2):
                scores[nz] = scores[nz] / m_norm[nz]
            return scores
    except Exception:
        pass

    # Dense path
    q = np.asarray(query_vec).astype("float32")
    M = np.asarray(matrix).astype("float32")
    # L2 normalize
    qn = np.linalg.norm(q) + 1e-8
    q = q / qn
    mn = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
    M = M / mn
    return (M @ q.reshape(-1, 1)).ravel()


def search(query: str, k: int = 5, **kwargs) -> Dict[str, Any]:
    """
    Basic TF‑IDF search.
    Accepts extra kwargs (mode, alpha, etc.) to be compatible with HTTP callers.
    Returns: {"query":..., "k":..., "hits":[{"i":idx,"score":float,"text":str,"meta":dict}, ...]}
    """
    if not isinstance(query, str) or not query.strip():
        return {"query": query, "k": k, "hits": []}

    tf = _load_tfidf()
    vec, nn = _get_vec_nn(tf)

    # Build query vector
    try:
        qv = vec.transform([query])
    except Exception:
        # Some vectorizers expose a different API; last resort
        if hasattr(vec, "encode"):
            qv = vec.encode([query])
        else:
            raise

    # Cosine similarity
    scores = _cosine_sim(qv, nn)

    if scores.size == 0:
        return {"query": query, "k": k, "hits": []}

    # Top-k
    k = max(1, int(k))
    k = min(k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    top = idx[np.argsort(scores[idx])[::-1]]

    texts = tf.get("texts", [])
    metas = tf.get("metas", [])
    hits = []
    for i in top:
        rec = {
            "i": int(i),
            "score": float(scores[i]),
            "text": texts[i] if i < len(texts) else "",
            "meta": metas[i] if i < len(metas) else {},
        }
        hits.append(rec)

    return {"query": query, "k": k, "hits": hits}


# ------------------------------- Health helpers -------------------------------

def health() -> Dict[str, Any]:
    try:
        tf = _load_tfidf()
        vec, nn = _get_vec_nn(tf)
        return {"ok": True, "docs": int(getattr(nn, "shape", (0,))[0])}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------- CLI -----------------------------------------

if __name__ == "__main__":
    print("[retriever] index dir:", _index_dir())
    print("[retriever] health:", health())
    print("[retriever] test:", search("cobalt nanowire synthesis", k=3))