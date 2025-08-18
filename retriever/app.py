from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal
import pickle, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

app = FastAPI(title="Nanochem Retriever (Hybrid)", version="1.1.0")

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    mode: Literal["tfidf","embed","hybrid"] = "hybrid"
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

_store = {"tfidf": None, "embed": None}

def _load_tfidf():
    if _store["tfidf"] is None:
        p = Path("index/tfidf.pkl")
        if not p.exists():
            raise RuntimeError("TF-IDF index missing. Build with index_jsonl.py")
        with open(p, "rb") as f:
            _store["tfidf"] = pickle.load(f)
    return _store["tfidf"]

def _load_embed():
    p = Path("index/embed.pkl")
    if not p.exists():
        return None
    if _store["embed"] is None:
        with open(p, "rb") as f:
            _store["embed"] = pickle.load(f)
    return _store["embed"]

def _embed_query(q: str, backend: str, model: str):
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model=model, input=[q])
        v = np.array(resp.data[0].embedding, dtype="float32")
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model)
        v = m.encode([q], normalize_embeddings=True)[0].astype("float32")
    # L2 norm
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

@app.get("/health")
def health():
    have_embed = Path("index/embed.pkl").exists()
    return {"status":"ok", "embeddings": have_embed}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    tf = _load_tfidf()
    vec = tf["vectorizer"]; X = tf["matrix"]
    metas = tf["metas"]; texts = tf["texts"]

    # TF-IDF sims
    qv = vec.transform([req.query])
    sims_tfidf = cosine_similarity(qv, X)[0]

    if req.mode == "tfidf":
        sims = sims_tfidf
    else:
        emb_store = _load_embed()
        if emb_store is None:
            sims = sims_tfidf  # fallback
        else:
            backend = emb_store["backend"]; model = emb_store["model"]
            E = emb_store["embeddings"]  # (N, d), L2-normalized
            q_emb = _embed_query(req.query, backend, model)
            sims_emb = (E @ q_emb)  # cosine since L2-normalized
            if req.mode == "embed":
                sims = sims_emb
            else:
                # Hybrid score
                sims = req.alpha * sims_emb + (1.0 - req.alpha) * sims_tfidf

    idxs = np.argsort(-sims)[: req.k]
    hits = []
    for i in idxs:
        m = metas[i]
        hits.append(SearchHit(
            score=float(sims[i]),
            text=texts[i],
            paper_id=str(m.get("paper_id","")),
            title=m.get("title",""),
            url=m.get("url"),
            license=m.get("license"),
            ents=m.get("ents"),
            links=m.get("links")
        ))
    return SearchResponse(hits=hits)
