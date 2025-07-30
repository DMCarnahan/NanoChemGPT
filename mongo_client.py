import os
from urllib.parse import quote_plus
from pymongo import MongoClient

_client = None
_db = None

def _computed_mongo_url() -> str:
    url = os.getenv("MONGO_URL") or os.getenv("MONGODB_URI")
    if url:
        return url
    host = os.getenv("MONGOHOST", "localhost")
    port = os.getenv("MONGOPORT", "27017")
    user = os.getenv("MONGOUSER")
    pwd  = os.getenv("MONGOPASSWORD")
    if user and pwd:
        return f"mongodb://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/?authSource=admin"
    return f"mongodb://{host}:{port}"

def get_db():
    global _client, _db
    if _db is not None:
        return _db
    uri = _computed_mongo_url()
    _client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    name = os.getenv("MONGO_DB", "nanochem")
    _db = _client[name]
    try:
        _db.uploads.create_index("ts")
        _db.uploads.create_index("filename")
        _db.qa.create_index([("created_at", 1)])
        _db.qa.create_index([("question", "text")], default_language="english")
        _db.parsed.create_index([("created_at", 1)])
    except Exception as e:
        print("[mongo] index creation warning:", e)
    return _db

def ping():
    return get_db().command("ping")
