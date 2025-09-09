from __future__ import annotations

import os, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np

# ------------------------------ Path resolution ------------------------------

def _env_paths() -> Dict[str, Path]:
    out: Dict[str, Path] = {}

    # 1) Explicit multi-path
    mp = os.getenv("RETRIEVER_INDEX_DIRS")
    if mp:
        parts = [p for p in mp.split(":") if p.strip()]
        for i, p in enumerate(parts):
            lab = "doc" if ("doc" in p) else ("passage" if "passage" in p else f"idx{i+1}")
            out[lab] = Path(p).resolve()
        return out

    # 2) Explicit singles
    doc = os.getenv("RETRIEVER_INDEX_DIR_DOC")
    pas = os.getenv("RETRIEVER_INDEX_DIR_PASSAGE")
    if doc: out["doc"] = Path(doc).resolve()
    if pas: out["passage"] = Path(pas).resolve()
    if out:
        return out

    # 3) Legacy single
    one = os.getenv("RETRIEVER_INDEX_DIR")
    if one:
        out["doc"] = Path(one).resolve()
        return out

    # 4) Heuristic defaults near this file
    base = Path(__file__).parent
    if (base / "index_doc").exists():
        out["doc"] = (base / "index_doc").resolve()
    if (base / "index_passage").exists():
        out["passage"] = (base / "index_passage").resolve()
    if out:
        return out

    out["doc"] = (base / "index").resolve()
    return out

def _labels_and_paths() -> List[Tuple[str, Path]]:
    env = _env_paths()
    items = list(env.items())
    # prefer doc then passage
    order = {"doc": 0, "passage": 1}
    items.sort(key=lambda kv: order.get(kv[0], 2))
    return items

# ------------------------------ Caches ---------------------------------------

_BUNDLES: Dict[Path, Dict[str, Any]] = {}
_VECS: Dict[Path, Tuple[Any, Any]] = {}

def reload_caches() -> bool:
    _BUNDLES.clear()
    _VECS.clear()
    return True

# ------------------------------ Utils ----------------------------------------
def _pick_from(obj, keys):
    """
    Return the first present value for keys without triggering truthiness
    on NumPy/SciPy objects. Works for np.load npz dicts and normal dicts.
    """
    # np.load(...) returns an NpzFile with `.files` listing keys
    if hasattr(obj, "files"):
        for k in keys:
            if k in obj.files:
                return obj[k]
        return None
    # Fallback: normal mapping
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
        # tolerate dict-like with .get
        try:
            v = obj.get(k)
        except Exception:
            v = None
        if v is not None:
            return v
    return None

def _load_rows_jsonl(idx: Path):
    rows_path = idx / "rows.jsonl"
    texts, metas = [], []
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if isinstance(r, dict):
                    texts.append(r.get("text", "") or "")
                    metas.append({k: r.get(k) for k in ("id","title","doi","url","pdf_url","oa_url","year","authors") if k in r})
    return texts, metas

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
                        i = int(k)
                        if 0 <= i < n:
                            arr[i] = v
                    x = arr
                else:
                    x = [x] * max(1, n)
            except Exception:
                x = [x] * max(1, n)
        if not isinstance(x, list):
            x = [x] * max(1, n)
        if len(x) < n:
            x = x + [fill() for _ in range(n - len(x))]
        elif len(x) > n:
            x = x[:n]
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

def _vectorizer_from_vocab_obj(vobj):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return None
    if isinstance(vobj, dict):
        vocab_map = vobj
    elif isinstance(vobj, list):
        vocab_map = {t: i for i, t in enumerate(vobj)}
    else:
        return None
    return TfidfVectorizer(vocabulary=vocab_map)

# ------------------------------ Loaders --------------------------------------

def _load_tfidf_for(idx: Path, force: bool = False) -> Dict[str, Any]:
    if not isinstance(idx, Path): idx = Path(idx)
    idx = idx.resolve()
    if idx in _BUNDLES and not force: return _BUNDLES[idx]

    pkl   = idx / "tfidf.pkl"
    npz   = idx / "tfidf.npz"
    vecj  = idx / "vectorizer.joblib"
    vocab = idx / "vocab.json"

    print(f"[retriever] Using INDEX_DIR={idx}")
    prefer_npz = os.getenv("RETRIEVER_PREFER_NPZ", "1").lower() in {"1", "true", "yes"}
    disable_pkl = os.getenv("RETRIEVER_DISABLE_PICKLE", "").lower() in {"1", "true", "yes"}

    # --- NPZ first if preferred ---
    if prefer_npz and npz.exists():
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
            except Exception as e:
                print(f"[retriever] vectorizer.joblib load failed: {e}")
                vectorizer = None

        if vectorizer is None and vocab.exists():
            try:
                vobj = json.loads(vocab.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                vobj = None
            vectorizer = _vectorizer_from_vocab_obj(vobj)

        if vectorizer is not None:
            bundle = {"kind": "matrix", "matrix": X, "vectorizer": vectorizer}
            texts, metas = _load_rows_jsonl(idx)
            if texts: bundle["texts"] = texts
            if metas: bundle["metas"] = metas
            try:
                n_cols = int(getattr(X, "shape", (0,0))[1]) if X is not None else 0
                vocab_size = int(len(getattr(vectorizer, "vocabulary_", {}) or {}))
                if vocab_size and n_cols and vocab_size != n_cols:
                    print(f"[retriever] WARN: matrix_cols={n_cols} != vocab_size={vocab_size} @ {idx}")
            except Exception as _e:
                print(f"[retriever] shape check warn: {_e}")

            _BUNDLES[idx] = _ensure_texts_metas(bundle)
            return _BUNDLES[idx]

    # --- PKL path (unless disabled) ---
    if (not disable_pkl) and pkl.exists():
        obj = None
        try:
            obj = joblib.load(pkl)
        except Exception as e:
            print(f"[retriever] Failed to load {pkl}: {e}. Will try npz/vectorizer fallbacks.")
            obj = None

        # dict format
        if isinstance(obj, dict):
            X = _pick_from(obj, ("matrix", "X", "tfidf"))
            vectorizer = _pick_from(obj, ("vectorizer", "tfidf_vectorizer"))
            # try sidecar vectorizer
            if X is not None and vectorizer is None and vecj.exists():
                try:
                    vectorizer = joblib.load(vecj)
                except Exception as e:
                    print(f"[retriever] vectorizer.joblib load failed: {e}")
                    vectorizer = None
            # rebuild from in-pickle vocabulary
            if X is not None and vectorizer is None:
                vobj = obj.get("vocabulary") or obj.get("vocab")
                if vobj is not None:
                    vectorizer = _vectorizer_from_vocab_obj(vobj)

            if X is not None and vectorizer is not None:
                bundle = {"kind": "matrix", "matrix": X, "vectorizer": vectorizer}
                for k in ("texts", "metas", "rows", "license", "titles"):
                    if k in obj: bundle[k] = obj[k]
                try:
                    n_cols = int(getattr(X, "shape", (0,0))[1]) if X is not None else 0
                    vocab_size = int(len(getattr(vectorizer, "vocabulary_", {}) or {}))
                    if vocab_size and n_cols and vocab_size != n_cols:
                        print(f"[retriever] WARN: matrix_cols={n_cols} != vocab_size={vocab_size} @ {idx}")
                except Exception as _e:
                    print(f"[retriever] shape check warn: {_e}")
                _BUNDLES[idx] = _ensure_texts_metas(bundle)
                return _BUNDLES[idx]
            # else fall through

        # tuple/list format
        if isinstance(obj, (list, tuple)):
            X, vectorizer, texts, metas = _pick_from_sequence(obj)
            if X is not None and vectorizer is not None:
                bundle = {"kind":"matrix","matrix": X, "vectorizer": vectorizer}
                if texts is not None: bundle["texts"] = texts
                if metas is not None: bundle["metas"] = metas
                try:
                    n_cols = int(getattr(X, "shape", (0,0))[1]) if X is not None else 0
                    vocab_size = int(len(getattr(vectorizer, "vocabulary_", {}) or {}))
                    if vocab_size and n_cols and vocab_size != n_cols:
                        print(f"[retriever] WARN: matrix_cols={n_cols} != vocab_size={vocab_size} @ {idx}")
                except Exception as _e:
                    print(f"[retriever] shape check warn: {_e}")
                _BUNDLES[idx] = _ensure_texts_metas(bundle)
                return _BUNDLES[idx]

    # --- NPZ second chance (if not preferred initially) ---
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
            except Exception as e:
                print(f"[retriever] vectorizer.joblib load failed: {e}")
                vectorizer = None

        if vectorizer is None and vocab.exists():
            try:
                vobj = json.loads(vocab.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                vobj = None
            vectorizer = _vectorizer_from_vocab_obj(vobj)

        if vectorizer is None:
            raise RuntimeError(f"Found {npz} but no vectorizer.joblib or vocab.json")

        bundle = {"kind":"matrix","matrix": X, "vectorizer": vectorizer}
        texts, metas = _load_rows_jsonl(idx)
        if texts: bundle["texts"] = texts
        if metas: bundle["metas"] = metas
        _BUNDLES[idx] = _ensure_texts_metas(bundle)
        return _BUNDLES[idx]

    raise RuntimeError(
        f"No TF-IDF index found or unreadable in {idx}. "
        f"Expected tfidf.pkl (dict/tuple) or tfidf.npz + vectorizer.joblib/vocab.json."
    )

def _get_vec_nn(idx: Path) -> Tuple[Any, Any]:
    if idx in _VECS: return _VECS[idx]
    tf = _load_tfidf_for(idx)
    vec = tf.get("vectorizer"); nn = tf.get("matrix")
    _VECS[idx] = (vec, nn); return _VECS[idx]

def _build_query_vec(vec, query: str):
    try:
        return vec.transform([query])
    except Exception:
        if hasattr(vec, "encode"):
            return vec.encode([query])
        raise

def _cosine_sim(qv, M):
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception:
        sp = None

    # 1) Dense 1D query vector
    if sp is not None and sp.issparse(qv):
        q = qv.toarray().ravel().astype("float32", copy=False)
    else:
        q = np.asarray(qv).ravel().astype("float32", copy=False)
    qn = np.linalg.norm(q) + 1e-8
    q = q / qn

    # 2) Sparse matrix path
    if sp is not None and sp.issparse(M):
        tmp = M @ q.reshape(-1, 1)          # this is usually a dense ndarray
        if sp.issparse(tmp):
            s = tmp.toarray().ravel()
        else:
            s = np.asarray(tmp).ravel()
        # row-norms for cosine (||row||). Use sparse ops to avoid densifying M.
        row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A.ravel()
        nz = row_norms > 0
        if nz.any():
            s[nz] = s[nz] / row_norms[nz]
        return s.astype("float32", copy=False)

    # 3) Dense matrix path
    Md = np.asarray(M, dtype="float32")
    row_norms = np.linalg.norm(Md, axis=1) + 1e-8
    return (Md @ q.reshape(-1, 1)).ravel() / row_norms

def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k)); k = min(k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]

def _safe_float(v, default):
    try: return float(v)
    except Exception: return float(default)

# ------------------------------ Public API -----------------------------------

def search(query: str, k: int = 5, **kwargs) -> Dict[str, Any]:
    merged: List[Dict[str, Any]] = []

    level = (kwargs.get("level") or os.getenv("RETRIEVER_LEVEL") or "doc").lower()
    label_paths = _labels_and_paths()
    if not label_paths:
        return {"query": query, "k": k, "hits": []}

    want_both = (level in ("both","all")) or (level == "auto" and len(label_paths) > 1)
    if want_both:
        targets: List[Tuple[str, Path]] = label_paths
    else:
        targets = [next(((l,p) for l,p in label_paths if l==level), label_paths[0])]

    k_doc = int(kwargs.get("k_doc", k))
    k_pas = int(kwargs.get("k_passage", k))
    k_other = int(max(1, k))
    w_doc = _safe_float(kwargs.get("w_doc", os.getenv("WEIGHT_DOC", 0.6)), 0.6)
    w_pas = _safe_float(kwargs.get("w_passage", os.getenv("WEIGHT_PASSAGE", 0.4)), 0.4)

    for lab, idx_path in targets:
        tf = _load_tfidf_for(idx_path)
        vec, nn = _get_vec_nn(idx_path)
        qv = _build_query_vec(vec, query)
        scores = _cosine_sim(qv, nn)
        if scores.size == 0: 
            continue
        kk = k_doc if lab == "doc" else (k_pas if lab == "passage" else k_other)
        top = _topk(scores, kk)
        texts = tf.get("texts") or []
        metas = tf.get("metas") or [{}] * len(texts)

        s = scores[top].astype("float32")
        if s.size > 0:
            s_min, s_max = float(np.min(s)), float(np.max(s))
            s = (s - s_min) / (s_max - s_min) if s_max > s_min else np.full_like(s, 0.5)

        weight = w_doc if lab == "doc" else (w_pas if lab == "passage" else 0.5)
        for i, sc in zip(top, s):
            i = int(i)
            meta = metas[i] if i < len(metas) else {}
            txt = (texts[i] if i < len(texts) else "") or ""
            merged.append({
                "i": i,
                "score": float(sc * weight),
                "text": txt[:1200],
                "meta": meta,
                "level": lab,
                "index_dir": str(idx_path),
            })

    if not merged:
        return {"query": query, "k": k, "hits": []}

    merged.sort(key=lambda h: h["score"], reverse=True)
    merged = merged[:max(1, int(k))]
    return {"query": query, "k": k, "level": level, "hits": merged, "levels": [lv for lv,_ in label_paths]}

def health() -> Dict[str, Any]:
    info = {"ok": True, "indexes": []}
    for lab, p in _labels_and_paths():
        try:
            tf = _load_tfidf_for(p)
            nn = tf.get("matrix"); vec = tf.get("vectorizer")
            n_rows, n_cols = (getattr(nn, "shape", (0,0)) or (0,0))
            vocab_size = int(len(getattr(vec, "vocabulary_", {}) or {}))
            info["indexes"].append({
                "label": lab, "path": str(p),
                "docs": int(n_rows),
                "matrix_cols": int(n_cols),
                "vocab_size": vocab_size
            })
        except Exception as e:
            info["indexes"].append({"label": lab, "path": str(p), "error": str(e)})
    if not info["indexes"]:
        return {"ok": False, "error": "no index dirs found"}
    return info

# -------- Back-compat shim for old /reload handlers --------

def _load_tfidf(force: bool = False) -> List[Dict[str, Any]]:
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
    print("[retriever] paths:", _env_paths())
    print("[retriever] health:", json.dumps(health(), indent=2))
    print("[retriever] doc:", json.dumps(search("cobalt nanowire synthesis", k=3, level="doc"), indent=2))
    print("[retriever] passage:", json.dumps(search("cobalt nanowire synthesis", k=3, level="passage"), indent=2))
    print("[retriever] both:", json.dumps(search("cobalt nanowire synthesis", k=5, level="both"), indent=2))
