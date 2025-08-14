"""
Mechanistic retriever:
- Builds a vector index over (system, method, parameters->effects, mechanism) strings
- Abstracts embedding backend so you can plug in OpenAI or sentence-transformers
"""
from __future__ import annotations
import pathlib, json, os
from typing import List, Dict, Callable, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
KB_JSONL = ROOT / "mechanistic_kb" / "mechanistic.jsonl"
INDEX_DIR = ROOT / "mechanistic_kb" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

def default_text_fn(e: Dict) -> str:
    lines = [e.get("system",""), e.get("synthesis_method","")]
    for mech in e.get("mechanisms", []):
        lines.append(f"MECH:{mech.get('name','')} EVID:{mech.get('evidence_snippet','')}")
    for p in e.get("parameters", []):
        lines.append(f"PARAM:{p.get('name','')} ROLE:{p.get('role','')}")
        for eff in p.get("effects", []):
            lines.append(f"EFFECT {eff.get('target','')} {eff.get('direction','')} {eff.get('mechanistic_rationale','')}")
    return "\n".join(lines)

def load_entries() -> List[Dict]:
    if not KB_JSONL.exists():
        return []
    out = []
    with KB_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            out.append(json.loads(line))
    return out

def build_corpus(entries: List[Dict], text_fn: Callable[[Dict], str]=default_text_fn) -> Tuple[List[str], List[Dict]]:
    texts, meta = [], []
    for e in entries:
        texts.append(text_fn(e))
        meta.append({"id": e["id"], "system": e.get("system",""), "method": e.get("synthesis_method","")})
    return texts, meta

# Embedding abstraction
class Embedder:
    """
    Local encoder using sentence-transformers.
    """
    def __init__(self, backend: str = "st", model: str | None = None, batch_size: int = 64):
        self.backend = backend
        self.model_id = model or os.getenv("ST_MODEL", "sentence-transformers/all-mpnet-base-v2")
        from sentence_transformers import SentenceTransformer
        # Lazy-load once
        self.model = SentenceTransformer(self.model_id, device="cpu")
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> List[List[float]]:
        # sentence-transformers returns np.ndarray; convert to py lists
        embs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=False, convert_to_numpy=True)
        return embs.tolist()

def build_index(embedder: Embedder):
    entries = load_entries()
    texts, meta = build_corpus(entries)
    if not texts:
        return None, []
    X = embedder.encode(texts)
    import numpy as np
    X = np.array(X, dtype="float32")
    if faiss is None:
        # Fallback: brute-force cosine similarity in Python
        return {"embeddings": X, "meta": meta, "texts": texts}, meta
    index = faiss.IndexFlatIP(X.shape[1])
    # normalize for cosine
    faiss.normalize_L2(X)
    index.add(X)
    faiss.write_index(index, str(INDEX_DIR / "mechanistic.faiss"))
    with (INDEX_DIR / "meta.json").open("w") as f:
        json.dump({"meta": meta, "texts": texts}, f)
    return index, meta

def search(query: str, k: int = 5, embedder: Embedder | None = None):
    embedder = embedder or Embedder()
    entries = load_entries()
    texts, meta = build_corpus(entries)
    if not texts:
        return []
    import numpy as np
    X = np.array(embedder.encode(texts), dtype="float32")
    qv = np.array(embedder.encode([query]), dtype="float32")
    if faiss is not None:
        faiss.normalize_L2(X); faiss.normalize_L2(qv)
        index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
        sims, idx = index.search(qv, k)
        idx = idx[0].tolist()
    else:
        # cosine via numpy
        def cos(a,b): 
            na = (a**2).sum()**0.5; nb=(b**2).sum()**0.5
            return float((a@b)/(na*nb+1e-12))
        sims = [cos(qv[0], X[i]) for i in range(len(X))]
        idx = list(sorted(range(len(X)), key=lambda i: -sims[i]))[:k]
    return [{"entry": entries[i], "text": texts[i], "meta": meta[i]} for i in idx]

if __name__ == "__main__":
    idx, meta = build_index(Embedder())
    print("Index built.", "entries=", len(meta))
