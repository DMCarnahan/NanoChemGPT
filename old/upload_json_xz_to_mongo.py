import os, sys, json, lzma, hashlib, datetime
from pymongo import MongoClient, UpdateOne

MONGO_URL = os.environ["MONGO_URL"]
DB  = os.environ.get("MONGO_DB") or MongoClient(MONGO_URL).get_default_database().name
COL = os.environ.get("BUILTIN_COLLECTION", "builtin_docs")
PATH = sys.argv[1]   # solid-state_dataset_20200713.json.xz

client = MongoClient(MONGO_URL)
db = client[DB]
col = db[COL]
col.create_index("hash", unique=True)

def hash_doc(d: dict) -> str:
    # stable content hash to avoid duplicates
    return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def iter_json_xz(path):
    with lzma.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        # sniff: array or lines
        buf = f.read(1)
        if not buf:
            return
        if buf == "[":
            yield from json.load(f)  # remainder of array
        else:
            # first char was part of first line; rewind into a small buffer
            first = buf + f.read()
            for line in first.splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)

def to_text(d: dict) -> str:
    # create a text field for embedding: concat all string leaves
    out = []
    def walk(x):
        if isinstance(x, str): out.append(x)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
    walk(d)
    return "\n".join(out)[:200_000]  # cap to avoid huge docs

batch, n, BATCH = [], 0, 500
for doc in iter_json_xz(PATH):
    h = hash_doc(doc)
    doc["_uploaded_at"] = datetime.datetime.utcnow()
    doc["hash"] = h
    if "text" not in doc:
        doc["text"] = to_text(doc)
    batch.append(UpdateOne({"hash": h}, {"$set": doc}, upsert=True))
    if len(batch) >= BATCH:
        col.bulk_write(batch, ordered=False)
        n += len(batch); print("upserted", n); batch.clear()
if batch:
    col.bulk_write(batch, ordered=False)
    n += len(batch); print("upserted", n)

print("done. total:", n)
