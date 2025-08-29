from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from app import app as flask_app

try:
    from retriever.api import api as retriever_api
except Exception:
    retriever_api = None

app = FastAPI(title="NanoChemGPT ASGI")

@app.get("/healthz")
def healthz():
    return {"ok": True}

if retriever_api is not None:
    app.mount("/retriever", retriever_api)

app.mount("/", WSGIMiddleware(flask_app))
