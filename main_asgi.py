from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.responses import RedirectResponse

from app import app as flask_app
from retriever.api import app as retr_app

app = FastAPI(docs_url="/_docs", redoc_url="/_redoc", openapi_url="/_openapi.json")

# retriever first
app.mount("/retriever", retr_app)

# Flask UI under /app
app.mount("/app", WSGIMiddleware(flask_app))

# redirect root to Flask UI
@app.get("/", include_in_schema=False)
def _root():
    return RedirectResponse(url="/app")
