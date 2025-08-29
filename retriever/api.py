from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .retriever import search as _search, health as _health  # in-process calls

api = FastAPI(title="Retriever API", version="1.0")

class SearchReq(BaseModel):
    query: str
    k: int = 5
    mode: Optional[str] = None   # ignored but accepted for compatibility
    alpha: Optional[float] = None

@api.get("/healthz")
def healthz():
    return _health()

@api.post("/search")
def search(req: SearchReq):
    return _search(query=req.query, k=req.k)
