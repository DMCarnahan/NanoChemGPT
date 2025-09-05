from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

from app import app as flask_app
from retriever.api import app as retr_app

# Keep FastAPI docs available
app = FastAPI(docs_url="/_docs", redoc_url="/_redoc", openapi_url="/_openapi.json")

# Mount retriever first so it isn't shadowed
app.mount("/retriever", retr_app)

# Mount Flask app under /app (explicit) and also at / (legacy/root)
app.mount("/app", WSGIMiddleware(flask_app))
app.mount("/", WSGIMiddleware(flask_app))
