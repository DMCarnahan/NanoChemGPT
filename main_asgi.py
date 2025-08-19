from asgiref.wsgi import WsgiToAsgi
from fastapi import FastAPI

from app import app as flask_app                 # Flask web app (WSGI)
from retriever.service import app as retr_app    # FastAPI retriever (ASGI)

root = FastAPI(title="NanoChemGPT (combined)")

# Mount retriever at /retriever
root.mount("/retriever", retr_app)

# Wrap Flask (WSGI) → ASGI and mount at /
root.mount("/", WsgiToAsgi(flask_app))

# Final ASGI app entrypoint:
app = root
