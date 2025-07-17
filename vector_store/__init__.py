"""Persistent vector store that
* embeds text with **intfloat/e5‑small‑v2** (384‑D, CPU‑friendly)
* stores vectors in FAISS `IndexFlatIP` + stable IDs
* **tags** every chunk as either ``builtin`` or ``upload``
  · builtin vectors are permanent
  · upload vectors expire automatically after 30 minutes **or** on demand via
    `clear_uploads()`
"""
from __future__ import annotations

import gzip, json, pathlib, threading, time, csv
from typing import List, Dict, Any

import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
from PyPDF2 import PdfReader

# ── config ────────────────────────────────────────────────────────────────
DATA_DIR   = pathlib.Path("data"); DATA_DIR.mkdir(exist_ok=True)
INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE  = DATA_DIR / "meta.json"   # JSON list[dict]

TOKENIZER   = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKS  = 300
DIM         = 384
UPLOAD_TTL  = 30 * 60          # 30 minutes in seconds

# ── embedding model (lazy) ────────────────────────────────────────────────
_embedder: SentenceTransformer | None = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("[vector_store] loading e5-small-v2 …")
        # keep kwargs minimal so we work with ST <2.6 as well
        _embedder = SentenceTransformer("intfloat/e5-small-v2")
    return _embedder

# ── helpers ───────────────────────────────────────────────────────────────

def _chunk(text: str) -> List[str]:
    words, buf, out = text.split(), [], []
    for w in words:
        buf.append(w)
        if len(TOKENIZER.encode(" ".join(buf))) >= CHUNK_TOKS:
            out.append(" ".join(buf)); buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def _embed_passages(txts: list[str]) -> np.ndarray:
    vecs = _get_embedder().encode(
        [f"passage: {t}" for t in txts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=16,
    ).astype("float32")
    return vecs


def _embed_query(q: str) -> np.ndarray:
    return _get_embedder().encode(
        f"query: {q}", normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

# flatten JSON / CSV -------------------------------------------------------

def _json_to_lines(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _json_to_lines(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _json_to_lines(v, f"{prefix}{i}.")
    else:
        yield f"{prefix[:-1]}: {obj}"

# ── index init / persistence ──────────────────────────────────────────────

def _new_index() -> faiss.IndexIDMap2:
    return faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))

if INDEX_FILE.exists():
    _index: faiss.IndexIDMap2 = faiss.read_index(str(INDEX_FILE))  # type: ignore[arg-type]
    _meta: List[Dict[str, Any]] = json.loads(META_FILE.read_text())
else:
    _index, _meta = _new_index(), []

_lock = threading.Lock()


def _persist():
    faiss.write_index(_index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(_meta))

# ── core public API ───────────────────────────────────────────────────────

def add_to_store(doc: str, *, tag: str = "upload", ts: float | None = None):
    if not doc:
        return
    ts = ts or time.time()
    chunks = _chunk(doc)
    vecs   = _embed_passages(chunks)
    with _lock:
        start = len(_meta)
        ids   = np.arange(start, start + len(chunks)).astype("int64")
        _index.add_with_ids(vecs, ids)
        _meta.extend({"text": c, "tag": tag, "ts": ts} for c in chunks)
        _persist()
        print(f"[vector_store] indexed {len(chunks)} chunks (total {len(_meta)})")


def add_json_bytes(b: bytes, *, tag="upload"):
    try:
        data = json.loads(b)
    except ValueError as e:
        print("JSON parse error:", e); return
    add_to_store("\n".join(_json_to_lines(data)), tag=tag)


def add_csv_bytes(b: bytes, *, delimiter=",", tag="upload"):
    lines = []
    for i, row in enumerate(csv.DictReader(b.decode().splitlines(), delimiter=delimiter)):
        for k, v in row.items():
            lines.append(f"row[{i}].{k}: {v}")
    add_to_store("\n".join(lines), tag=tag)

# ── search with auto‑expiry ───────────────────────────────────────────────

def _expire_old_uploads():
    now = time.time()
    keep = [rec for rec in _meta if rec["tag"] == "builtin" or now - rec["ts"] < UPLOAD_TTL]
    if len(keep) == len(_meta):
        return
    print("[vector_store] expiring", len(_meta) - len(keep), "upload chunks …")
    _rebuild_index(keep)


def _rebuild_index(records: List[dict]):
    global _index, _meta
    _index = _new_index(); _meta = []
    for rec in records:
        add_to_store(rec["text"], tag=rec["tag"], ts=rec["ts"])  # rebuild path
    print("[vector_store] index rebuilt")


def search(query: str, k: int = 4) -> str:
    _expire_old_uploads()
    if _index.ntotal == 0 or not query:
        return ""
    qvec = _embed_query(query)
    with _lock:
        scores, ids = _index.search(qvec, min(k, _index.ntotal))
    hits = [_meta[i]["text"] for i in ids[0] if i != -1]
    return "\n\n".join(hits)

# ── management helpers ─────────────────────────────────────────────────---

def clear_uploads():
    keep = [rec for rec in _meta if rec["tag"] == "builtin"]
    _rebuild_index(keep)
    _persist()
    print("[vector_store] cleared all uploaded chunks")


def stats():
    return {"chunks": len(_meta), "vectors": int(_index.ntotal)}

# ── builtin corpus load (tag=builtin) ─────────────────────────────────────
BUILTIN_DIR = pathlib.Path(__file__).parent / "builtin_data"


def _load_builtin_once():
    if any(rec["tag"] == "builtin" for rec in _meta):
        return
    for path in BUILTIN_DIR.rglob("*"):
        if path.suffix == ".json":
            add_json_bytes(path.read_bytes(), tag="builtin")
        elif path.suffix in {".csv", ".tsv"}:
            add_csv_bytes(path.read_bytes(), delimiter="\t" if path.suffix == ".tsv" else ",", tag="builtin")
        elif path.suffix == ".gz" and path.name.endswith(".json.gz"):
            add_json_bytes(gzip.open(path, "rb").read(), tag="builtin")
        elif path.suffix == ".pdf":
            text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
            add_to_store(text, tag="builtin")
    print("[vector_store] builtin_data embedded")

# ── module init -----------------------------------------------------------
_load_builtin_once()

# ── CLI -------------------------------------------------------------------
if __name__ == "__main__":
    print(stats())
