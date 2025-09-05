# main_asgi.py
from fastapi import FastAPI, Request
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.responses import RedirectResponse
from app import app as flask_app
from retriever.api import app as retr_app

app = FastAPI(docs_url="/_docs", redoc_url="/_redoc", openapi_url="/_openapi.json")

# 1) mount retriever first
app.mount("/retriever", retr_app)

# 2) mount Flask under /app
app.mount("/app", WSGIMiddleware(flask_app))

# 3) redirect root → /app (so the UI loads)
@app.get("/", include_in_schema=False)
async def _root():
    return RedirectResponse(url="/app")

# 4) legacy routes → 307 to /app/*
def _redir(path: str):
    return RedirectResponse(url=f"/app{path}", status_code=307)

@app.api_route("/ask", methods=["GET","POST"], include_in_schema=False)
async def _ask_legacy(_: Request):   return _redir("/ask")

@app.api_route("/parse", methods=["POST"], include_in_schema=False)
async def _parse_legacy(_: Request): return _redir("/parse")

@app.api_route("/upload", methods=["POST"], include_in_schema=False)
async def _upload_legacy(_: Request): return _redir("/upload")

@app.api_route("/api/history", methods=["GET"], include_in_schema=False)
async def _hist_legacy(_: Request):  return _redir("/api/history")

@app.api_route("/api/history/{id}", methods=["GET"], include_in_schema=False)
async def _hist_id_legacy(id: str):  return _redir(f"/api/history/{id}")

@app.api_route("/api/uploads", methods=["GET"], include_in_schema=False)
async def _uploads_legacy(_: Request): return _redir("/api/uploads")

@app.api_route("/status/{jid}", methods=["GET"], include_in_schema=False)
async def _status_legacy(jid: str):  return _redir(f"/status/{jid}")

@app.api_route("/upload_builtin", methods=["POST"], include_in_schema=False)
async def _up_built_legacy(_: Request): return _redir("/upload_builtin")

@app.api_route("/clear_uploads", methods=["POST"], include_in_schema=False)
async def _clr_up_legacy(_: Request): return _redir("/clear_uploads")

@app.api_route("/save_txt", methods=["POST"], include_in_schema=False)
async def _save_txt_legacy(_: Request): return _redir("/save_txt")

@app.api_route("/parse_upload", methods=["POST"], include_in_schema=False)
async def _parse_up_legacy(_: Request): return _redir("/parse_upload")
