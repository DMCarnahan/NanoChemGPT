from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
import os, pickle, json, logging, numpy as np
from pathlib import Path
from functools import lru_cache

# ---------- Paths ----------
BASE_DIR   = Path(__file__).resolve().parent
INDEX_DIR  = Path(os.getenv("RETRIEVER_INDEX_DIR", BASE_DIR / "index")).resolve()
TFIDF_PATH = (INDEX_DIR / "tfidf.pkl").resolve()
EMBED_PATH = (INDEX_DIR / "embed.pkl").resolve()
_TFIDF = {}

app = FastAPI(title="Nanochem Retriever (Hybrid)", version="1.2.0")

logger = logging.getLogger(__name__)
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

# ---------- Loaders ----------
@lru_cache(maxsize=1)

def _load_tfidf():
    try:
        V = load("retriever/index/vectorizer.joblib")
        X = sparse.load_npz("retriever/index/tfidf.npz")
        ids = json.loads(Path("retriever/index/meta.json").read_text())["ids"]
        _TFIDF.update(dict(vec=V, X=X, ids=ids))
        logger.info("[retriever] TF-IDF loaded: docs=%d, terms=%d", X.shape[0], X.shape[1])
    except Exception as e:
        logger.warning("[retriever] TF-IDF not available: %s", e)

def tfidf_search(q: str, k: int = 8):
    if not _TFIDF:
        _load_tfidf()
    if not _TFIDF:
        return []
    qv = _TFIDF["vec"].transform([q])
    scores = (qv @ _TFIDF["X"].T).toarray()[0]
    idx = np.argsort(scores)[::-1][:k]
    return [(_TFIDF["ids"][i], float(scores[i])) for i in idx]

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