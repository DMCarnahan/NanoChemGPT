from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback

from retriever import retriever as R  

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    mode: str | None = None
    alpha: float | None = None

@app.post("/retriever/search")
def search(req: SearchRequest):
    try:
        return R.search(query=req.query, k=req.k)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"retriever.search error: {e}")

@app.get("/retriever/healthz")
def healthz():
    try:
        b = R._load_tfidf()
        X = b.get("matrix")
        n_docs = int(getattr(X, "shape", [0, 0])[0]) if X is not None else 0
        n_terms = int(getattr(X, "shape", [0, 0])[1]) if X is not None else 0
        return {"ok": True, "docs": n_docs, "terms": n_terms}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/retriever/reload")
def reload():
    """
    Clear in-process caches so the next search() uses the freshly rebuilt index.
    Optionally 'warm' the cache by loading once.
    """
    try:
        R.reload_caches()
        b = R._load_tfidf()
        X = b.get("matrix")
        n_docs = int(getattr(X, "shape", [0, 0])[0]) if X is not None else 0
        n_terms = int(getattr(X, "shape", [0, 0])[1]) if X is not None else 0
        return {"ok": True, "docs": n_docs, "terms": n_terms}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"reload error: {e}")