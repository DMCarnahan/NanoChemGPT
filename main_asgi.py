from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.wsgi import WSGIMiddleware

# Flask app (WSGI)
from app import app as flask_app

# retriever as FastAPI 
try:
    from retriever.retriever import app as retriever_api
except Exception:
    retriever_api = None

app = FastAPI(title="NanoChemGPT", version="1.0")

@app.get("/healthz")
def healthz():
    return {"ok": True}

if retriever_api is not None:
    app.mount("/retriever", retriever_api)
app.mount("/", WSGIMiddleware(flask_app))