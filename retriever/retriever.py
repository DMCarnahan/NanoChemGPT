from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
import os, pickle, numpy as np
from pathlib import Path
from functools import lru_cache
import joblib
from scipy.sparse import load_npz

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = []

# 1) Explicit env var takes precedence
env_dir = os.getenv("RETRIEVER_INDEX_DIR")
if env_dir:
    CANDIDATES.append(Path(env_dir))

# 2) Common project layouts
CANDIDATES += [
    BASE_DIR / "retriever" / "index",    
    BASE_DIR / "index",                   
    BASE_DIR.parent / "retriever" / "index"  
]
INDEX_DIR = Path(os.getenv("RETRIEVER_INDEX_DIR") or (Path(__file__).parent / "index")).resolve()
TFIDF_PATH = Path(os.getenv("TFIDF_PATH") or (INDEX_DIR / "tfidf.pkl"))
EMBED_PATH = (INDEX_DIR / "embed.pkl").resolve()

print(f"[retriever] Using INDEX_DIR={INDEX_DIR}")

app = FastAPI(title="Nanochem Retriever (Hybrid)", version="1.2.0")

# ---------- Schemas ----------
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    mode: Literal["tfidf", "embed", "hybrid"] = "hybrid"
    alpha: float = 0.7  # weight for embeddings in hybrid

class SearchHit(BaseModel):
    score: float
    text: str
    paper_id: str
    title: str
    url: Optional[str] = None
    license: Optional[dict] = None
    ents: Optional[list] = None
    links: Optional[list] = None

class SearchResponse(BaseModel):
    hits: List[SearchHit]


class Embedder:
    """
    Minimal wrapper so legacy code importing `Embedder` keeps working.
    It uses your existing _load_embed() config and backs off to
    sentence-transformers if no embedding store is on disk.
    """
    def __init__(self, backend: str | None = None, model: str | None = None):
        store = _load_embed()
        self.backend = backend or (store and store.get("backend")) or "sentence-transformers"
        self.model = model or (store and store.get("model")) or "sentence-transformers/all-MiniLM-L6-v2"
        self._st = None  # lazy SentenceTransformer

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "openai":
            from openai import OpenAI  
            client = OpenAI()
            resp = client.embeddings.create(model=self.model, input=texts)
            arr = np.array([d.embedding for d in resp.data], dtype="float32")
            arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
            return arr
        else:
            from sentence_transformers import SentenceTransformer
            if self._st is None:
                self._st = SentenceTransformer(self.model)
            vecs = self._st.encode(texts, normalize_embeddings=True)
            return vecs.astype("float32")

    def embed(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

# ---------- Loaders ----------
@lru_cache(maxsize=1)
def _load_tfidf() -> Dict[str, Any]:
    """Load the TF-IDF index from disk."""
    print(f"[retriever] TFIDF_PATH={TFIDF_PATH} exists={TFIDF_PATH.exists()}")
    if not TFIDF_PATH.exists():
        raise RuntimeError(
            f"TF-IDF index missing at {TFIDF_PATH}. "
            f"Build it or set RETRIEVER_INDEX_DIR to the folder containing tfidf.pkl."
        )
    with open(TFIDF_PATH, "rb") as f:
        store = pickle.load(f)

    # Layout A: matrix style
    if {"vectorizer", "matrix"} <= set(store.keys()):
        vec   = store["vectorizer"]
        X     = store["matrix"]
        metas = store.get("metas")
        texts = store.get("texts")
        if metas is None or texts is None:
            rows = store.get("rows", [])
            if rows and not texts:
                texts = [r.get("text", "") for r in rows]
            if rows and not metas:
                metas = rows
        return {"kind": "matrix", "vectorizer": vec, "matrix": X, "metas": metas, "texts": texts}

    # Layout B: NN style
    if {"rows", "vec", "nn"} <= set(store.keys()):
        rows = store["rows"]
        vec  = store["vec"]
        nn   = store["nn"]
        texts = [r.get("text", "") for r in rows]
        metas = rows
        return {"kind": "nn", "rows": rows, "vec": vec, "nn": nn, "metas": metas, "texts": texts}

    raise RuntimeError(f"Unrecognized TF-IDF format keys: {list(store.keys())}")

@lru_cache(maxsize=1)
def _load_embed():
    """Load the embedding index from disk."""
    if not EMBED_PATH.exists():
        return None
    with open(EMBED_PATH, "rb") as f:
        return pickle.load(f)  # expects {"backend","model","embeddings","metas","texts"} or similar

def _embed_query(q: str, backend: str, model: str) -> np.ndarray:
    """Embed a query string using the specified backend and model."""
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model=model, input=[q])
        v = np.array(resp.data[0].embedding, dtype="float32")
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model)
        v = m.encode([q], normalize_embeddings=True)[0].astype("float32")
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def _index_dir() -> Path:
    return Path(os.getenv("RETRIEVER_INDEX_DIR") or (Path(__file__).parent / "index")).resolve()

def _load_tfidf():
    idx = _index_dir()
    pkl  = idx / "tfidf.pkl"
    npz  = idx / "tfidf.npz"
    vecj = idx / "vectorizer.joblib"
    vocab = idx / "vocab.json"

    print(f"[retriever] Using INDEX_DIR={idx}")

    # 1) New format: single pickle with vectorizer + matrix
    if pkl.exists():
        obj = joblib.load(pkl)
        X = obj.get("matrix") or obj.get("X")
        vectorizer = obj.get("vectorizer")
        if X is None or vectorizer is None:
            raise RuntimeError(f"Malformed {pkl}: expected dict with 'matrix' and 'vectorizer'")
        return {"kind": "sklearn_pkl", "matrix": X, "vectorizer": vectorizer}

    # 2) Legacy format: tfidf.npz + vectorizer.joblib (preferred)
    if npz.exists():
        X = load_npz(npz)
        if vecj.exists():
            vectorizer = joblib.load(vecj)
            return {"kind": "sklearn_npz", "matrix": X, "vectorizer": vectorizer}
        # 3) Last resort: vocabulary-only (works but idf_ may be missing)
        if vocab.exists():
            from sklearn.feature_extraction.text import TfidfVectorizer
            vocab_list = json.loads(vocab.read_text(encoding="utf-8"))
            vocab_map = {t: i for i, t in enumerate(vocab_list)}
            vectorizer = TfidfVectorizer(vocabulary=vocab_map)
            return {"kind": "vocab_only", "matrix": X, "vectorizer": vectorizer}

    raise RuntimeError(
        f"TF-IDF index missing: expected {pkl} or {npz} (+ {vecj} / {vocab}). "
        f"Set RETRIEVER_INDEX_DIR to the folder containing your index."
    )
# ---------- API ----------
@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "tfidf_path": str(TFIDF_PATH),
        "tfidf_exists": TFIDF_PATH.exists(),
        "embed_path": str(EMBED_PATH),
        "embed_exists": EMBED_PATH.exists()
    }

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search the corpus using TF-IDF, embeddings, or hybrid."""
    tf = _load_tfidf()
    kind = tf["kind"]

    # --- TF-IDF similarity ----
    if kind == "matrix":
        from sklearn.metrics.pairwise import cosine_similarity
        vec, X = tf["vectorizer"], tf["matrix"]
        qv = vec.transform([req.query])
        sims_tfidf = cosine_similarity(qv, X)[0]
    else:
        # NearestNeighbors index
        vec, nn = tf["vec"], tf["nn"]
        qv = vec.transform([req.query])
        dist, idx = nn.kneighbors(qv, n_neighbors=min(req.k, len(tf["texts"])))
        sims_nn = 1.0 - dist[0]  # convert distances to similarity
        sims_tfidf = np.zeros(len(tf["texts"]), dtype="float32")
        sims_tfidf[idx[0]] = sims_nn

    sims = sims_tfidf

    # --- Optional embeddings / hybrid ---
    if req.mode in ("embed", "hybrid"):
        emb_store = _load_embed()
        if emb_store is not None:
            backend = emb_store["backend"]
            model = emb_store["model"]
            E = emb_store["embeddings"]  # (N,d) L2-normalized
            q_emb = _embed_query(req.query, backend, model)
            sims_emb = (E @ q_emb)
            sims = sims_emb if req.mode == "embed" else (req.alpha * sims_emb + (1.0 - req.alpha) * sims_tfidf)

    k = max(1, min(req.k, len(tf["texts"])))
    idxs = np.argsort(-sims)[:k]

    hits: List[SearchHit] = []
    metas, texts = tf["metas"], tf["texts"]
    for i in idxs:
        m = metas[i] if isinstance(metas, list) else {}
        hits.append(SearchHit(
            score=float(sims[i]),
            text=texts[i],
            paper_id=str(m.get("paper_id", "")),
            title=m.get("title", ""),
            url=m.get("url"),
            license=m.get("license"),
            ents=m.get("ents"),
            links=m.get("links")
        ))
    return SearchResponse(hits=hits)

@app.post("/reload")
def reload_indexes():
    """Reload the TF-IDF and embedding indexes from disk."""
    _load_tfidf.cache_clear()
    _load_embed.cache_clear()
    _ = _load_tfidf()
    return {"reloaded": True, "tfidf_path": str(TFIDF_PATH)}