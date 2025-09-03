# retriever/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback

# Import the retriever implementation module
from retriever import retriever as R  

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    k: int = 8

@app.post("/search")
def search(req: SearchRequest):
    try:
        return R.search(query=req.query, k=req.k)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"retriever.search error: {e}")

@app.get("/healthz")
def healthz():
    try:
        b = R._load_tfidf()
        X = b.get("matrix")
        docs = int(getattr(X, "shape", [0, 0])[0]) if X is not None else 0
        terms = int(getattr(X, "shape", [0, 0])[1]) if X is not None else 0
        return {"ok": True, "docs": docs, "terms": terms}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/reload")
def reload():
    try:
        R.reload_caches()          
        b = R._load_tfidf()        
        X = b.get("matrix")
        docs = int(getattr(X, "shape", [0, 0])[0]) if X is not None else 0
        terms = int(getattr(X, "shape", [0, 0])[1]) if X is not None else 0
        return {"ok": True, "docs": docs, "terms": terms}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"reload error: {e}")
