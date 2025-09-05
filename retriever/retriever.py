from __future__ import annotations

import os, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

# -------------------------------- Path resolution --------------------------------

def _env_paths() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    # Multi-path first
    mp = os.getenv("RETRIEVER_INDEX_DIRS")
    if mp:
        parts = [p for p in mp.split(":") if p.strip()]
        for i, p in enumerate(parts):
            label = "doc" if ("doc" in p or i == 0) else ("passage" if "passage" in p else f"idx{i+1}")
            out[label] = Path(p).resolve()
        return out
    # Explicit singles
    doc = os.getenv("RETRIEVER_INDEX_DIR_DOC")
    pas = os.getenv("RETRIEVER_INDEX_DIR_PASSAGE")
    if doc or pas:
        if doc: out["doc"] = Path(doc).resolve()
        if pas: out["passage"] = Path(pas).resolve()
        return out
    # Legacy single
    one = os.getenv("RETRIEVER_INDEX_DIR")
    if one:
        out["doc"] = Path(one).resolve()
        return out
    # Heuristic local defaults: prefer index_doc/index_passage if present
    base = Path(__file__).parent
    docp = (base / "index_doc").resolve()
    pasp = (base / "index_passage").resolve()
    if docp.exists():
        out["doc"] = docp
        if pasp.exists():
            out["passage"] = pasp
        return out
    # fallback to ./index
    out["doc"] = (base / "index").resolve()
    return out

# ------------------------------- Index loading -----------------------------------

_BUNDLES: Dict[Path, Dict[str, Any]] = {}
_VECS: Dict[Path, Tuple[Any, Any]] = {}

def reload_caches() -> bool:
    _BUNDLES.clear(); _VECS.clear(); return True

def _ensure_texts_metas(bundle: dict) -> dict:
    X = bundle.get("matrix")
    n = int(getattr(X, "shape", (0, 0))[0]) if X is not None else 0
    texts = bundle.get("texts")
    metas = bundle.get("metas")
    rows  = bundle.get("rows")

    if rows and not texts:
        try:
            texts = [r.get("text", "") for r in rows if isinstance(r, dict)]
        except Exception:
            pass
    if rows and not metas:
        metas = rows

    def _as_list(x, n, fill):
        if x is None: return [fill() for _ in range(n)]
        if isinstance(x, tuple): x = list(x)
        if isinstance(x, dict):
            try:
                if all(str(k).isdigit() for k in x.keys()):
                    arr = [fill() for _ in range(n)]
                    for k, v in x.items():
                        i = int(k); 
                        if 0 <= i < n: arr[i] = v
                    x = arr
                else:
                    x = [x] * max(1, n)
            except Exception:
                x = [x] * max(1, n)
        if not isinstance(x, list): x = [x] * max(1, n)
        if len(x) < n: x = x + [fill() for _ in range(n - len(x))]
        elif len(x) > n: x = x[:n]
        return x

    texts = _as_list(texts, n, lambda: "")
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    metas = _as_list(metas, n, lambda: {})
    metas = [m if isinstance(m, dict) else {"meta": m} for m in metas]

    bundle["texts"] = texts
    bundle["metas"] = metas
    bundle["vec"] = bundle.get("vectorizer")
    bundle["nn"]  = bundle.get("matrix")
    return bundle

def _pick_from_sequence(obj):
    """Heuristically pick (matrix, vectorizer, texts, metas) from a tuple/list joblib dump."""
    mat = vec = None; texts = metas = None
    for x in obj:
        try:
            if hasattr(x, "shape"):
                mat = x
            elif hasattr(x, "transform"):
                vec = x
            elif isinstance(x, list) and x and isinstance(x[0], str):
                texts = x
            elif isinstance(x, list) and x and isinstance(x[0], dict):
                metas = x
        except Exception:
            pass
    return mat, vec, texts, metas

def _load_tfidf_for(idx: Path, force: bool = False) -> Dict[str, Any]:
    if not isinstance(idx, Path): idx = Path(idx)
    idx = idx.resolve()
    if idx in _BUNDLES and not force: return _BUNDLES[idx]

    pkl  = idx / "tfidf.pkl"
    npz  = idx / "tfidf.npz"
    vecj = idx / "vectorizer.joblib"
    vocab= idx / "vocab.json"

    print(f"[retriever] Using INDEX_DIR={idx}")

    # 1) Preferred: tfidf.pkl
    if pkl.exists():
        obj = joblib.load(pkl)
        # dict format (expected)
        if isinstance(obj, dict):
            X = obj.get("matrix") or obj.get("X") or obj.get("tfidf")
            vectorizer = obj.get("vectorizer") or obj.get("vec")
            bundle = {"kind": "matrix", "matrix": X, "vectorizer": vectorizer}
            for k in ("texts","metas","rows","license","titles"):
                if k in obj: bundle[k] = obj[k]
            # if either X or vectorizer is missing, try to fall back to npz+vectorizer.joblib
            if X is None or vectorizer is None:
                # fall through to npz path
                pass
            else:
                _BUNDLES[idx] = _ensure_texts_metas(bundle)
                return _BUNDLES[idx]
        # tuple/list format
        if isinstance(obj, (list, tuple)):
            X, vectorizer, texts, metas = _pick_from_sequence(obj)
            if X is not None and vectorizer is not None:
                bundle = {"kind":"matrix","matrix": X, "vectorizer": vectorizer}
                if texts is not None: bundle["texts"] = texts
                if metas is not None: bundle["metas"] = metas
                _BUNDLES[idx] = _ensure_texts_metas(bundle)
                return _BUNDLES[idx]
        # unknown object; continue to npz fallbacks

    # 2) npz + vectorizer.joblib
    if npz.exists():
        try:
            from scipy.sparse import load_npz  # type: ignore
            X = load_npz(npz)
        except Exception:
            f = np.load(npz)
            key = "arr_0" if "arr_0" in f.files else next(iter(f.files))
            X = f[key]
        vectorizer = None
        if vecj.exists():
            try:
                vectorizer = joblib.load(vecj)
            except Exception:
                vectorizer = None

        # legacy vocab.json
        if vectorizer is None and vocab.exists():
            from sklearn.feature_extraction.text import TfidfVectorizer
            vocab_list = json.loads(vocab.read_text(encoding="utf-8", errors="ignore"))
            vocab_map  = {t: i for i, t in enumerate(vocab_list)}
            vectorizer = TfidfVectorizer(vocabulary=vocab_map)

        if vectorizer is None:
            raise RuntimeError(f"Found {npz} but no vectorizer.joblib or vocab.json")

        bundle = {"kind":"matrix","matrix": X, "vectorizer": vectorizer}
        # sidecars
        for sidecar in ("rows.pkl","rows.jsonl","texts.jsonl","meta.json"):
            p = idx / sidecar
            if not p.exists(): continue
            try:
                if p.suffix == ".pkl":
                    rows = joblib.load(p)
                    if isinstance(rows, list): bundle["rows"] = rows
                elif p.suffix == ".json":
                    bundle["metas"] = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                else:
                    rows = []
                    with p.open("r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            try: rows.append(json.loads(line))
                            except Exception: pass
                    if rows: bundle["rows"] = rows
            except Exception:
                pass
        _BUNDLES[idx] = _ensure_texts_metas(bundle)
        return _BUNDLES[idx]

    raise RuntimeError(f"No TF-IDF index found or unreadable in {idx}. Expected tfidf.pkl (dict/tuple) or tfidf.npz+vectorizer.joblib.")

def _get_vec_nn(idx: Path) -> Tuple[Any, Any]:
    if idx in _VECS: return _VECS[idx]
    tf = _load_tfidf_for(idx)
    vec = tf.get("vectorizer"); nn = tf.get("matrix")
    _VECS[idx] = (vec, nn); return _VECS[idx]

def _cosine_sim(query_vec, matrix):
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(matrix):
            q = query_vec
            if sp.issparse(q):
                qn = np.sqrt(q.multiply(q).sum())
                if qn > 0: q = q / qn
            scores = (matrix @ q.T).toarray().ravel()
            m_norm = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
            nz = (m_norm > 0)
            if not np.allclose(m_norm[nz], 1.0, atol=1e-2):
                scores[nz] = scores[nz] / m_norm[nz]
            return scores
    except Exception:
        pass
    q = np.asarray(query_vec).astype("float32")
    M = np.asarray(matrix).astype("float32")
    qn = np.linalg.norm(q) + 1e-8; q = q / qn
    mn = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8; M = M / mn
    return (M @ q.reshape(-1,1)).ravel()

# ------------------------------- Public API --------------------------------

def _labels_and_paths() -> List[Tuple[str, Path]]:
    env = _env_paths()
    items = list(env.items())
    items.sort(key=lambda kv: {"doc":0,"passage":1}.get(kv[0], 2))
    return items

def _build_query_vec(vec, query: str):
    try:
        return vec.transform([query])
    except Exception:
        if hasattr(vec, "encode"):
            return vec.encode([query])
        raise

def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k)); k = min(k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]

def search(query: str, k: int = 5, **kwargs) -> Dict[str, Any]:
    level = (kwargs.get("level") or os.getenv("RETRIEVER_LEVEL") or "doc").lower()
    label_paths = _labels_and_paths()
    if not label_paths: return {"query": query, "k": k, "hits": []}
    want_both = (level in ("both","all")) or (level == "auto" and len(label_paths) > 1)
    targets: List[Tuple[str, Path]] = label_paths if want_both else [next(((l,p) for l,p in label_paths if l==level), label_paths[0])]
    k_doc = int(kwargs.get("k_doc", k)); k_pas = int(kwargs.get("k_passage", k)); k_other = int(max(1,k))
    w_doc = float(kwargs.get("w_doc", os.getenv("WEIGHT_DOC", 0.6))); w_pas = float(kwargs.get("w_passage", os.getenv("WEIGHT_PASSAGE", 0.4)))
    merged: List[Dict[str, Any]] = []
    for lab, idx_path in targets:
        tf = _load_tfidf_for(idx_path); vec, nn = _get_vec_nn(idx_path)
        qv = _build_query_vec(vec, query); scores = _cosine_sim(qv, nn)
        if scores.size == 0: continue
        kk = k_doc if lab=="doc" else (k_pas if lab=="passage" else k_other)
        top = _topk(scores, kk); texts = tf.get("texts") or []; metas = tf.get("metas") or [{}]*len(texts)
        s = scores[top].astype("float32"); 
        if s.size>0:
            s_min, s_max = float(np.min(s)), float(np.max(s))
            s = (s - s_min)/(s_max - s_min) if s_max>s_min else np.full_like(s, 0.5)
        weight = w_doc if lab=="doc" else (w_pas if lab=="passage" else 0.5)
        for i, sc in zip(top, s):
            meta = metas[int(i)] if int(i)<len(metas) else {}; txt = (texts[int(i)] if int(i)<len(texts) else "") or ""
            merged.append({"i": int(i), "score": float(sc*weight), "text": txt[:1200], "meta": meta, "level": lab, "index_dir": str(idx_path)})
    if not merged: return {"query": query, "k": k, "hits": []}
    merged.sort(key=lambda h: h["score"], reverse=True); merged = merged[:max(1,int(k))]
    return {"query": query, "k": k, "level": level, "hits": merged, "levels": [lv for lv,_ in label_paths]}

def health() -> Dict[str, Any]:
    info = {"ok": True, "indexes": []}
    for lab, p in _labels_and_paths():
        try:
            tf = _load_tfidf_for(p); nn = tf.get("matrix"); docs = int(getattr(nn, "shape", (0,))[0])
            info["indexes"].append({"label": lab, "path": str(p), "docs": docs})
        except Exception as e:
            info["indexes"].append({"label": lab, "path": str(p), "error": str(e)})
    if not info["indexes"]: return {"ok": False, "error": "no index dirs found"}
    return info

# --------- Backward-compat shim for retriever.api.reload() ---------

def _load_tfidf(force: bool = False) -> List[Dict[str, Any]]:
    """
    Back-compat: warm all configured indexes into memory and return a summary.
    api.py used to call this when /retriever/reload is hit.
    """
    warmed = []
    for label, path in _labels_and_paths():
        try:
            tf = _load_tfidf_for(path, force=force)
            nn = tf.get("matrix")
            docs = int(getattr(nn, "shape", (0,))[0]) if nn is not None else 0
            warmed.append({"label": label, "path": str(path), "docs": docs})
        except Exception as e:
            warmed.append({"label": label, "path": str(path), "error": str(e)})
    return warmed

if __name__ == "__main__":
    print("[retriever] indexes:", _env_paths())
    print("[retriever] health:", json.dumps(health(), indent=2))
    print("[retriever] test DOC:", json.dumps(search("cobalt nanowire synthesis", k=3, level="doc"), indent=2))
    print("[retriever] test PASSAGE:", json.dumps(search("cobalt nanowire synthesis", k=3, level="passage"), indent=2))
    print("[retriever] test BOTH:", json.dumps(search("cobalt nanowire synthesis", k=5, level="both"), indent=2))
