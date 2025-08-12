from __future__ import annotations

import os, re, time, threading, gzip, json, pathlib, tempfile, contextlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

# ---------------- Boot log ----------------
print("[vector_store.v2] EMBED_BACKEND =", os.getenv("EMBED_BACKEND", "st"),
      "| EMBED_MODEL =", os.getenv("EMBED_MODEL", ""),
      "| OPENAI_EMB  =", os.getenv("EMBED_OPENAI_MODEL", ""))

# ---------------- Config ----------------
DATA_DIR = pathlib.Path(os.getenv("VECTORSTORE_DIR", "/tmp/index")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "index"; INDEX_DIR.mkdir(parents=True, exist_ok=True)

TTL_SEC         = int(os.getenv("UPLOAD_TTL_SEC", "1800"))    # 30 min
EMBED_BACKEND   = os.getenv("EMBED_BACKEND", "st").lower()    # "st" | "openai"
EMBED_MODEL     = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
OPENAI_EMB      = os.getenv("EMBED_OPENAI_MODEL", "text-embedding-3-small")
EMB_BATCH       = int(os.getenv("EMBED_BATCH", "64"))
DISABLE_PERSIST = os.getenv("VS_DISABLE_PERSIST", "0") == "1"
DEFER_EMBED     = os.getenv("DEFER_EMBED", "1") == "1"        # defer during preload by default

# Rerank & MMR knobs
DO_MMR          = os.getenv("VS_MMR", "1") == "1"
MMR_LAMBDA      = float(os.getenv("VS_MMR_LAMBDA", "0.5"))    # 0..1 (0=diverse,1=relevant)
DO_LLM_RERANK   = os.getenv("VS_LLM_RERANK", "0") == "1"      # default off
LLM_RERANK_TOP  = int(os.getenv("VS_LLM_RERANK_TOP", "16"))
LLM_RERANK_KEEP = int(os.getenv("VS_LLM_RERANK_KEEP", "8"))
LLM_RERANK_MODEL= os.getenv("VS_LLM_RERANK_MODEL", "gpt-4o-mini")

# Chunking
CHUNK_SIZE      = int(os.getenv("VS_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP   = int(os.getenv("VS_CHUNK_OVERLAP", "150"))

# ---------------- State ----------------
SCHEMA_VERSION = 2
_index: Optional[faiss.Index] = None
_meta: List[Dict[str, Any]] = []      # [{id, tag, ts, text, doc_id}]
_dirty_index = False                  
_lock = threading.Lock()

# ---------------- Lazy sentence-transformers ----------------
_ST_MODEL = None
def _load_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(EMBED_MODEL)
        print("[vector_store.v2] loaded sentence-transformers:", EMBED_MODEL)
    return _ST_MODEL

def _encode_st(texts: List[str]) -> np.ndarray:
    model = _load_st_model()
    emb = model.encode(
        texts,
        normalize_embeddings=True,      # unit vectors for cosine/IP
        show_progress_bar=False,
        batch_size=EMB_BATCH,
    )
    return np.asarray(emb, dtype="float32")

# ---------------- OpenAI embeddings ----------------
def _openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def _encode_openai(texts: List[str]) -> np.ndarray:
    client = _openai_client()
    out: List[List[float]] = []
    for i in range(0, len(texts), EMB_BATCH):
        chunk = texts[i:i+EMB_BATCH]
        # Basic retry loop
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=OPENAI_EMB, input=chunk)
                out.extend(e.embedding for e in resp.data)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
    arr = np.asarray(out, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / norms).astype("float32")

def _encode(texts: List[str]) -> np.ndarray:
    backend = os.getenv("EMBED_BACKEND", EMBED_BACKEND).lower()
    try:
        if backend == "openai":
            print("[vector_store.v2] using OpenAI embeddings:", OPENAI_EMB)
            return _encode_openai(texts)
        print("[vector_store.v2] using sentence-transformers embeddings:", EMBED_MODEL)
        return _encode_st(texts)
    except Exception as e:
        print("[vector_store.v2] embed failed:", repr(e),
              "— set EMBED_BACKEND=openai to avoid local torch.")
        raise

# ---------------- Index IO ----------------
def _index_path() -> pathlib.Path:
    return INDEX_DIR / "index.faiss"

def _meta_path() -> pathlib.Path:
    return INDEX_DIR / "meta.v2.json.gz"

def _reset_index(d: int):
    global _index
    _index = faiss.IndexFlatIP(d)

def _get_index(d: int) -> faiss.Index:
    global _index
    if _index is None:
        ipath = _index_path()
        if ipath.exists():
            try:
                _index = faiss.read_index(str(ipath))
                _load_meta()
                print("[vector_store.v2] loaded FAISS index with ntotal=", _index.ntotal)
                return _index
            except Exception as e:
                print("[vector_store.v2] read_index failed, rebuilding:", e)
        _reset_index(d)
    return _index

def _atomic_write(path: pathlib.Path, data: bytes):
    if DISABLE_PERSIST:
        return
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tf:
            tmp = pathlib.Path(tf.name)
            tf.write(data)
        tmp.replace(path)
    finally:
        with contextlib.suppress(Exception):
            if tmp and tmp.exists():
                tmp.unlink()

def _persist():
    if DISABLE_PERSIST:
        return
    try:
        if _index is not None:
            faiss.write_index(_index, str(_index_path()))
        payload = {"schema": SCHEMA_VERSION, "meta": _meta}
        b = gzip.compress(json.dumps(payload).encode("utf-8"))
        _atomic_write(_meta_path(), b)
    except Exception as e:
        print("[vector_store.v2] persist error:", e)

def _load_meta():
    global _meta
    p = _meta_path()
    if p.exists():
        try:
            with gzip.open(p, "rb") as f:
                payload = json.loads(f.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("schema") == SCHEMA_VERSION:
                _meta = payload.get("meta", [])
            else:
                # best effort legacy load
                with gzip.open(INDEX_DIR / "meta.json.gz", "rt", encoding="utf-8") as f2:
                    _meta = json.load(f2)
        except Exception:
            _meta = []

# ---------------- Chunking ----------------
def _split_paragraphs(text: str) -> List[str]:
    # Keep paragraphs (double-newline), but avoid tiny crumbs
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or [text]

def _chunk(text: str, *, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # Header-aware: try to split on headings first
    paras = _split_paragraphs(text)
    chunks: List[str] = []
    current = ""
    for p in paras:
        if current and len(current) + 1 + len(p) > size:
            chunks.append(current)
            current = p
        else:
            current = (current + "\n\n" + p).strip() if current else p
        # further split long paragraphs
        while len(current) > size:
            cut = size
            # back off to nearest sentence end
            m = re.search(r".{1,%d}[\.!?]" % (size - overlap), current)
            cut = m.end() if m else size - overlap
            chunks.append(current[:cut].strip()[:size])
            current = current[cut - overlap:].strip()
    if current:
        chunks.append(current.strip())
    # Final trim and cap
    return [c[:size] for c in chunks] or [text[:size]]

# ---------------- Rebuild ----------------
def _rebuild_index_locked():
    """(Re)embed all texts and rebuild FAISS. Caller must hold _lock."""
    global _dirty_index
    texts = [m["text"] for m in _meta]
    if not texts:
        _reset_index(1536)
        _persist()
        _dirty_index = False
        return
    embs = _encode(texts)
    d = embs.shape[1]
    _reset_index(d)
    _index.add(embs)
    _persist()
    _dirty_index = False
    print(f"[vector_store.v2] rebuilt index with {len(_meta)} chunks; dim={d}")

def _ensure_index():
    with _lock:
        if _index is None:
            ipath = _index_path()
            if ipath.exists():
                try:
                    _get_index(1536)  # dim ignored when reading
                except Exception:
                    pass
        if _index is None or _dirty_index or (_index.ntotal != len(_meta)):
            _rebuild_index_locked()

# ---------------- Helpers: Router & Shortlist ----------------
METHOD_TAGS = {
    "sol gel": "builtin:sol-gel", "sol-gel": "builtin:sol-gel",
    "solid state": "builtin:solid-state", "solid-state": "builtin:solid-state",
    "hydrothermal": "builtin:hydrothermal",
    "cvd": "builtin:cvd", "ald": "builtin:ald",
    "coprecipitation": "builtin:coprecipitation", "co-precipitation": "builtin:coprecipitation",
    "electrodeposition": "builtin:electrodeposition",
    "solvothermal": "builtin:solvothermal",
}
CHEM_REGEX = re.compile(
    r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*|Co(?:\d|O|\dO\d)?|Ni(?:\d|O|\dO\d)?|TiO2|PdCl2|Pd|Pt|Fe2O3|CuO|Al2O3|MOF|perovskite)\b"
)

def _route_query(q: str) -> Tuple[List[str], List[str]]:
    ql = q.lower()
    tags: List[str] = []
    for key, tag in METHOD_TAGS.items():
        if key in ql:
            tags.append(tag)
    mats = [m for m in CHEM_REGEX.findall(q) if m]
    common = re.findall(r"\b(cobalt|palladium|nickel|alumina|perovskite|mof|graphene)\b", ql)
    must_keywords = sorted(set([*mats, *common]), key=str.lower)

    if not tags:
        tags = ["builtin:", "upload:", "mongo:"]
    return tags, must_keywords

def _shortlist_indices(metas: List[Dict[str, Any]],
                       must_tags: List[str],
                       must_keywords: List[str],
                       cap: int = 200) -> List[int]:
    idxs = list(range(len(metas)))
    if must_tags:
        idxs = [i for i in idxs if any(str(metas[i].get("tag","")).startswith(t) for t in must_tags)]
    if must_keywords:
        kws = [kw.lower() for kw in must_keywords]
        filt = [i for i in idxs if all(kw in metas[i]["text"].lower() for kw in kws)]
        if filt:
            idxs = filt
    return idxs[:cap] if idxs else []

def _mmr_select(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int,
                lam: float = 0.5) -> List[int]:
    """Classic MMR on cosine/IP space. query_vec: (1,d), doc_vecs: (n,d)."""
    if doc_vecs.shape[0] <= k:
        return list(range(doc_vecs.shape[0]))
    sims = (doc_vecs @ query_vec.T).ravel()  # (n,)
    selected = []
    candidates = set(range(doc_vecs.shape[0]))
    while len(selected) < k and candidates:
        if not selected:
            i = int(np.argmax(sims))
            selected.append(i); candidates.remove(i); continue
        # diversity term: max similarity to any already selected doc
        div = np.max(doc_vecs[list(selected)] @ doc_vecs.T, axis=0)
        scores = lam * sims + (1 - lam) * (-div)
        scores[list(selected)] = -1e9
        i = int(np.argmax(scores))
        selected.append(i); candidates.remove(i)
    return selected

# LLM reranker
def _llm_rerank(query: str, items: List[Tuple[int,str,str]]) -> List[Tuple[int,str,str]]:
    """items: list of (meta_idx, tag, text). Returns reordered list."""
    if not DO_LLM_RERANK or not items:
        return items
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        sample = items[:LLM_RERANK_TOP]
        prompt = "Score each context 0-10 for answering the query. Return JSON list of {i,score}.\n"
        prompt += f"Query: {query}\n\n"
        for i,(mid, tag, txt) in enumerate(sample):
            prompt += f"[{i}] ({tag}) {txt[:800]}\n\n"
        resp = client.chat.completions.create(
            model=LLM_RERANK_MODEL,
            temperature=0,
            messages=[{"role":"user","content":prompt}]
        )
        text = resp.choices[0].message.content or "[]"
        import json as _json
        scores = {int(d["i"]): float(d["score"]) for d in _json.loads(text) if "i" in d and "score" in d}
        ordered = sorted(range(len(sample)), key=lambda i: scores.get(i, 0.0), reverse=True)
        reord = [sample[i] for i in ordered][:LLM_RERANK_KEEP]
        return reord + items[len(sample):]
    except Exception as e:
        print("[vector_store.v2] llm_rerank failed:", e)
        return items

# ---------------- Public API ----------------
def add_document(text: str, *, tag: str = "upload", doc_id: Optional[str] = None, defer_embed: Optional[bool] = None) -> List[int]:
    """
    Add a full document, chunk it, and append to the store.
    Returns the list of meta indices created.
    """
    if defer_embed is None:
        defer_embed = DEFER_EMBED
    chunks = _chunk(text)
    idxs = []
    with _lock:
        cur = len(_meta)
        ts = int(time.time())
        for i, c in enumerate(chunks):
            mid = cur + i
            _meta.append({"id": mid, "tag": tag, "ts": ts, "text": c, "doc_id": doc_id})
            idxs.append(mid)

        if defer_embed:
            global _dirty_index
            _dirty_index = True
            _persist()
        else:
            if _index is None or _index.ntotal != cur:
                _rebuild_index_locked()
            else:
                embs = _encode(chunks)
                _get_index(embs.shape[1]).add(embs)
                _persist()

        print(f"[vector_store.v2] indexed {len(chunks)} chunks (total {len(_meta)}) tag={tag} doc_id={doc_id}")
    return idxs

def add_to_store(text: str, tag: str = "upload", defer_embed: Optional[bool] = None):
    """Back-compat alias for single-document add."""
    add_document(text, tag=tag, doc_id=None, defer_embed=defer_embed)

def delete_by_tag(prefix: str) -> int:
    """Delete all chunks whose tag starts with prefix. Returns # removed."""
    with _lock:
        before = len(_meta)
        keep = [m for m in _meta if not str(m.get("tag","")).startswith(prefix)]
        removed = before - len(keep)
        if removed:
            _meta[:] = keep
            global _dirty_index
            _dirty_index = True
            _persist()
            print(f"[vector_store.v2] delete_by_tag('{prefix}') removed {removed} chunks")
        return removed

def delete_by_ids(ids: List[int]) -> int:
    """Delete specific meta ids. Returns # removed."""
    ids_set = set(int(i) for i in ids)
    with _lock:
        keep = [m for m in _meta if int(m.get("id",-1)) not in ids_set]
        removed = len(_meta) - len(keep)
        if removed:
            _meta[:] = keep
            global _dirty_index
            _dirty_index = True
            _persist()
            print(f"[vector_store.v2] delete_by_ids removed {removed} chunks")
        return removed

def clear_uploads():
    """Remove transient upload:* chunks and mark index dirty."""
    return delete_by_tag("upload:")

def list_meta(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Introspection helper for UI."""
    return _meta[offset:offset+limit]

def usage_summary() -> Dict[str, Any]:
    """Return quick stats for the debug panel."""
    tags = {}
    for m in _meta:
        t = str(m.get("tag",""))
        tags[t] = tags.get(t, 0) + 1
    return {
        "schema": SCHEMA_VERSION,
        "count_chunks": len(_meta),
        "count_tags": len(tags),
        "tags": sorted([(k, v) for k, v in tags.items()], key=lambda x: x[0]),
        "index_ntotal": (None if _index is None else _index.ntotal),
    }

def search(query: str, k: int = 8, *,
           must_tags: list[str] | None = None,
           must_keywords: list[str] | None = None,
           return_structured: bool = False) -> str | Dict[str, Any]:
    """
    Hybrid search with:
      - router-derived must_tags/must_keywords if not provided
      - lexical shortlist
      - dense re-rank
      - MMR and optional LLM rerank

    If return_structured=True, returns {"hits":[{id, tag, score, text}], "prompt": "..."}.
    Otherwise returns a prompt-ready string with explicit [SRC tag] prefixes.
    """
    _ensure_index()
    with _lock:
        if not _meta or _index is None or _index.ntotal == 0:
            return "" if not return_structured else {"hits": [], "prompt": ""}

        # 1) Route if caller didn't provide filters
        if not must_tags and not must_keywords:
            rt_tags, rt_kws = _route_query(query)
            must_tags = rt_tags
            must_keywords = rt_kws
            print(f"[vector_store.v2] router: tags={must_tags} kws={must_keywords}")

        # 2) Shortlist by tags/keywords
        cand_idxs = _shortlist_indices(_meta, must_tags or [], must_keywords or [], cap=200)
        if not cand_idxs:
            cand_idxs = list(range(len(_meta)))  # fallback

        # 3) Dense over shortlist (local temp index)
        texts = [_meta[i]["text"] for i in cand_idxs]
        qv = _encode([query])                     # (1,d)
        dv = _encode(texts)                       # (n,d)

        d = dv.shape[1]
        tmp = faiss.IndexFlatIP(d)
        tmp.add(dv)
        D, I = tmp.search(qv, min(max(k*2, k), len(texts)))  # a little wider

        # 4) Collect candidates
        cand = [(cand_idxs[j], _meta[cand_idxs[j]].get("tag","ctx"), texts[j], float(D[0][ii])) for ii, j in enumerate(I[0])]

        # 5) MMR 
        if DO_MMR and len(cand) > k:
            select = _mmr_select(qv.astype("float32"), dv[I[0]], k=min(len(cand), max(k, 8)), lam=MMR_LAMBDA)
            cand = [cand[i] for i in select]

        # 6) LLM rerank 
        cand_llm = [(mid, tag, txt) for (mid, tag, txt, _sc) in cand]
        cand_llm = _llm_rerank(query, cand_llm)
        if cand_llm is not None and len(cand_llm) >= 1 and len(cand_llm) <= len(cand):
            # align back with scores where possible
            lookup = {(mid, tag, txt): sc for (mid, tag, txt, sc) in cand}
            cand = [(mid, tag, txt, lookup.get((mid, tag, txt), 0.0)) for (mid, tag, txt) in cand_llm]

        # 7) Final slice
        cand = cand[:k]

        # 8) Emit
        if return_structured:
            hits = [{"id": mid, "tag": tag, "score": score, "text": txt} for (mid, tag, txt, score) in cand]
            prompt = "\n---\n".join([f"[SRC {h['tag']}] {h['text']}" for h in hits])
            return {"hits": hits, "prompt": prompt}
        else:
            out = [f"[SRC {tag}] {txt}" for (_mi, tag, txt, _sc) in cand]
            return "\n---\n".join(out)

# ---------------- Expirer (uploads TTL) ----------------
def _expire_uploads():
    while True:
        time.sleep(60)
        now = int(time.time())
        changed = False
        with _lock:
            keep = []
            for m in _meta:
                tag = str(m.get("tag",""))
                if tag.startswith("upload:") and now - int(m.get("ts", now)) > TTL_SEC:
                    changed = True
                    continue
                keep.append(m)
            if changed:
                _meta[:] = keep
                global _dirty_index
                _dirty_index = True
                _persist()
                print("[vector_store.v2] expired old uploads; marked index dirty")

threading.Thread(target=_expire_uploads, daemon=True).start()
