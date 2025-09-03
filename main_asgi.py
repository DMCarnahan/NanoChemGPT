from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

from app import app as flask_app              
from retriever.api import app as retr_app     

app = FastAPI()

# Mount the retriever FIRST so /retriever/* routes are handled there
app.mount("/retriever", retr_app)

# Then mount Flask at the root
app.mount("/", WSGIMiddleware(flask_app))
