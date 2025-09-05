# main_asgi.py
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

from app import app as flask_app
from retriever.api import app as retr_app

# Keep FastAPI docs available
app = FastAPI(docs_url="/_docs", redoc_url="/_redoc", openapi_url="/_openapi.json")

# 1) Mount retriever first so it wins over the catch-all root mount
app.mount("/retriever", retr_app)

# 2) Mount Flask under /app for explicit-prefixed access
app.mount("/app", WSGIMiddleware(flask_app))

# 3) ALSO mount Flask at "/" so legacy clients posting to /ask, /upload, etc. work
#    This avoids relying on 307 redirects for POST bodies and eliminates 404s.
app.mount("/", WSGIMiddleware(flask_app))
