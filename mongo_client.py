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

from pymongo.errors import OperationFailure

def get_db():
    global _client, _db
    if _db is not None:
        return _db
    uri = _computed_mongo_url()
    _client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    name = os.getenv("MONGO_DB", "nanochem")
    _db = _client[name]

    try:
        # Uploads
        _db.uploads.create_index("ts")
        _db.uploads.create_index("filename")

        # Q&A – keep existing text index; avoid OptionsConflict noise
        try:
            _db.qa.create_index([("created_at", 1)])
            # If you already have a text index on question, this might raise code 85 -> ignore
            _db.qa.create_index([("question", "text"), ("answer", "text")],
                                default_language="english",
                                name="qa_text")
        except OperationFailure as e:
            if getattr(e, "code", None) == 85:
                print("[mongo] qa text index exists (different options) – keeping existing.")
            else:
                print("[mongo] qa text index warning:", e)

        # Parsed – ensure a text index exists (this was missing in your logs)
        try:
            _db.parsed.create_index([("created_at", 1)])
            _db.parsed.create_index([("question", "text"), ("raw_text", "text")],
                                    default_language="english",
                                    name="parsed_text")
        except OperationFailure as e:
            if getattr(e, "code", None) == 85:
                print("[mongo] parsed text index exists – keeping existing.")
            else:
                print("[mongo] parsed text index warning:", e)

    except Exception as e:
        print("[mongo] index creation warning:", e)

    return _db

def fetch_parsed_context(q: str, limit: int = 2) -> str:
    try:
        db = get_db()
        try:
            cur = db.parsed.find({"raw_text": {"$regex": q, "$options": "i"}})
        except Exception:
            return ""
        items = list(cur.sort("created_at", -1).limit(limit))
    except Exception as e:
        print("[parsed_ctx] query failed:", e)
        return ""

    pieces = []
    for d in items:
        p = d.get("parsed") or {}
        hdr  = "; ".join(p.get("hardware", [])[:5])
        reag = "; ".join((r.get("description") for r in p.get("reagents", [])[:6] if isinstance(r, dict)))
        proc = "; ".join(p.get("procedure", [])[:6])
        parts = []
        if hdr:  parts.append(f"Hardware: {hdr}")
        if reag: parts.append(f"Materials: {reag}")
        if proc: parts.append(f"Procedure: {proc}")
        if parts:
            pieces.append(" • ".join(parts))
    return "\n".join(pieces)

def ping():
    return get_db().command("ping")
