from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List

# Import your retriever module
from . import retriever as R

app = FastAPI(title="Retriever API", version="1.0.0")

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    level: Optional[str] = None
    k_doc: Optional[int] = None
    k_passage: Optional[int] = None
    w_doc: Optional[float] = None
    w_passage: Optional[float] = None

@app.get("/health")
def health() -> Dict[str, Any]:
    return R.health()

@app.post("/reload")
def reload() -> Dict[str, Any]:
    try:
        R.reload_caches()
        warmed = R._load_tfidf(force=True)  # back-compat shim
        return {"ok": True, "warmed": warmed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    try:
        res = R.search(query=req.query, k=req.k,
                       level=req.level, k_doc=req.k_doc, k_passage=req.k_passage,
                       w_doc=req.w_doc, w_passage=req.w_passage)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
