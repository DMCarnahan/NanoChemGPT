from asgiref.wsgi import WsgiToAsgi
from fastapi import FastAPI
from app import app as flask_app
from retriever.service import app as retriever_app

root = FastAPI(title="NanoChemGPT (combined)")
root.mount("/retriever", retriever_app)          # FastAPI retriever at /retriever/*
root.mount("/", WsgiToAsgi(flask_app))          # Flask app at /
app = root
