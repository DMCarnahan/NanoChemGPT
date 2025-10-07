import os
import traceback
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import retriever as R

app = FastAPI(title="Retriever API", version="1.1.0-debug")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    R.reload_caches()
    warmed = []
    if hasattr(R, "_load_tfidf"):
        try:
            warmed = R._load_tfidf(force=True)  # type: ignore
        except Exception as e:
            warmed = [{"error": str(e)}]
    return {"ok": True, "warmed": warmed}


@app.get("/diag")
def diag() -> Dict[str, Any]:
    import os

    info = {
        "env": {
            "RETRIEVER_INDEX_DIRS": os.getenv("RETRIEVER_INDEX_DIRS"),
            "RETRIEVER_INDEX_DIR_DOC": os.getenv("RETRIEVER_INDEX_DIR_DOC"),
            "RETRIEVER_INDEX_DIR_PASSAGE": os.getenv("RETRIEVER_INDEX_DIR_PASSAGE"),
            "RETRIEVER_INDEX_DIR": os.getenv("RETRIEVER_INDEX_DIR"),
            "WEIGHT_DOC": os.getenv("WEIGHT_DOC"),
            "WEIGHT_PASSAGE": os.getenv("WEIGHT_PASSAGE"),
        },
        "indexes": [],
    }
    for label, p in getattr(R, "_labels_and_paths")():
        p = str(p)
        files = []
        for name in (
            "tfidf.pkl",
            "tfidf.npz",
            "vectorizer.joblib",
            "vocab.json",
            "rows.pkl",
            "rows.jsonl",
            "texts.jsonl",
            "meta.json",
        ):
            files.append(
                {"name": name, "exists": os.path.exists(os.path.join(p, name))}
            )
        info["indexes"].append({"label": label, "path": p, "files": files})
    return info


@app.post("/search")
def search(req: SearchRequest, request: Request):
    try:
        return R.search(
            query=req.query,
            k=req.k,
            level=req.level,
            k_doc=req.k_doc,
            k_passage=req.k_passage,
            w_doc=req.w_doc,
            w_passage=req.w_passage,
        )
    except Exception as e:
        # Lazy auto-build + graceful degradation path for missing TF-IDF index
        missing_msg = "No TF-IDF index found" in str(e)
        if missing_msg:
            try:
                # Attempt to build indexes for each configured label path
                from .retriever import _labels_and_paths, _ensure_tfidf_index

                built_any = False
                for _lab, _p in _labels_and_paths():
                    before = (_p / "tfidf.pkl").exists() or (_p / "tfidf.npz").exists()
                    _ensure_tfidf_index(_p)
                    after = (_p / "tfidf.pkl").exists() or (_p / "tfidf.npz").exists()
                    built_any = built_any or (after and not before)
                if built_any:
                    # Retry once after attempted build
                    try:
                        return R.search(
                            query=req.query,
                            k=req.k,
                            level=req.level,
                            k_doc=req.k_doc,
                            k_passage=req.k_passage,
                            w_doc=req.w_doc,
                            w_passage=req.w_passage,
                        )
                    except Exception:  # fall through to generic handling
                        pass
            except Exception as auto_e:  # building attempt failed
                logger.warning("[retriever][auto-build] attempt failed: %s", auto_e)

            # Return a 200 with advisory instead of 500
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "query": req.query,
                    "hits": [],
                    "warning": "tfidf_index_missing",
                    "detail": (
                        "TF-IDF index missing or unreadable; build with "
                        "python retriever/index_jsonl.py --bundle <bundle.jsonl> --index_dir retriever/index_doc"
                    ),
                },
            )
        debug = (
            request.query_params.get("debug") == "1"
            or request.headers.get("X-Debug") == "1"
            or os.getenv("RETRIEVER_DEBUG", "").lower() in {"1", "true", "yes"}
        )
        tb = "".join(traceback.format_exc())

    # Always log the error so it appears in container logs
        print(f"[retriever][error] {e}\n{tb}")

        if debug:
            # Return JSON with traceback for visibility from any client
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": str(e),
                    "traceback": tb,
                    "paths": [
                        (lab, str(p)) for lab, p in getattr(R, "_labels_and_paths")()
                    ],
                    "env": {
                        "RETRIEVER_INDEX_DIRS": os.getenv("RETRIEVER_INDEX_DIRS"),
                        "RETRIEVER_INDEX_DIR_DOC": os.getenv("RETRIEVER_INDEX_DIR_DOC"),
                        "RETRIEVER_INDEX_DIR_PASSAGE": os.getenv(
                            "RETRIEVER_INDEX_DIR_PASSAGE"
                        ),
                        "RETRIEVER_INDEX_DIR": os.getenv("RETRIEVER_INDEX_DIR"),
                    },
                },
            )

        # Generic detail to avoid leaking internals in prod
        raise HTTPException(status_code=500, detail="retriever_search_failed")
