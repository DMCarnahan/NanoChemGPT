from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.responses import RedirectResponse

from app import app as flask_app
from retriever.api import app as retr_app

app = FastAPI()  # keep default docs at /docs

# retriever first
app.mount("/retriever", retr_app)

# put Flask under /app
app.mount("/app", WSGIMiddleware(flask_app))

# optional: redirect / → /app
@app.get("/", include_in_schema=False)
def _root():
    return RedirectResponse(url="/app")
