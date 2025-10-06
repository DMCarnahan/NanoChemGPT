from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _pick_text(rec: Dict, prefer_key: Optional[str] = None) -> Optional[str]:
    if prefer_key and rec.get(prefer_key):
        return str(rec[prefer_key]).strip()
    for k in ("text", "content", "abstract", "body", "methods", "title"):
        v = rec.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _pick_id(rec: Dict, fallback_idx: int) -> str:
    for k in ("id", "uid", "doc_id", "hash", "doi"):
        v = rec.get(k)
        if v:
            return str(v)
    return f"rec:{fallback_idx}"


def _load_texts(
    src: Path, limit: Optional[int], text_key: Optional[str]
) -> tuple[List[str], List[str]]:
    ids, texts = [], []
    for i, rec in enumerate(_iter_jsonl(src)):
        if limit and len(texts) >= limit:
            break
        t = _pick_text(rec, text_key)
        if not t:
            continue
        ids.append(_pick_id(rec, i))
        texts.append(t)
    return ids, texts


def _embed_openai(texts: List[str], model: str) -> np.ndarray:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. `pip install openai`") from e
    client = OpenAI()
    vecs: List[np.ndarray] = []
    B = 256
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        resp = client.embeddings.create(model=model, input=chunk)
        for d in resp.data:
            vecs.append(np.array(d.embedding, dtype="float32"))
    return np.vstack(vecs)


def _embed_st(
    texts: List[str],
    model: str,
    batch_size: int = 64,
    device: Optional[str] = None,  # e.g. "cuda" / "cpu"
    normalize: bool = True,
) -> np.ndarray:
    """
    Works with:
      • ST-packaged models: "sentence-transformers/allenai-specter", "sentence-transformers/allenai-specter", ...
      • Plain HF checkpoints: "pranav-s/MaterialsBERT", "m3rg-iitd/matscibert"
    """
    try:
        from sentence_transformers import SentenceTransformer, models
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers not installed. `pip install sentence-transformers`"
        ) from e

    # Try loading as a ready-made ST model first
    try:
        st = SentenceTransformer(
            model or "sentence-transformers/allenai-specter", device=device
        )
    except Exception:
        # Fallback: build an ST pipeline from a plain HF model (adds mean pooling)
        tr = models.Transformer(model, max_seq_length=512)  # BERT-base length
        pool = models.Pooling(
            tr.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,  # mean pooling usually best for sentences
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        st = SentenceTransformer(modules=[tr, pool], device=device)

    emb = st.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,  # cosine-ready if True
    )
    return emb.astype("float32")


def _build_faiss(emb: np.ndarray, metric: str, out_path: Path):
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "faiss-cpu not installed. Try: `pip install faiss-cpu`"
        ) from e
    dim = emb.shape[1]
    if metric.lower() == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric.lower() == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("metric must be 'ip' or 'l2'")
    index.add(emb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    return index


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Build a FAISS index from a JSONL KB.")
    p.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Input JSONL file (one record per line).",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output FAISS index file, e.g. data/index.faiss",
    )
    p.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="Optional meta JSON path (defaults to OUT with .meta.json)",
    )
    p.add_argument(
        "--text-key",
        type=str,
        default=None,
        help="Preferred key for text (else auto: text/content/abstract/body/title)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records for indexing (debug)",
    )
    p.add_argument(
        "--backend",
        type=str,
        default=os.getenv("EMBED_BACKEND", "st"),
        help="st | openai",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model. For openai, defaults to text-embedding-3-small; for st, allenai-specter.",
    )
    p.add_argument("--metric", type=str, default="ip", help="FAISS metric: ip or l2")
    args = p.parse_args(argv)

    meta_out = args.meta_out or args.out.with_suffix(".meta.json")

    ids, texts = _load_texts(args.src, args.limit, args.text_key)
    if not texts:
        print(
            f"[build] No texts found in {args.src}. Check keys or file.",
            file=sys.stderr,
        )
        sys.exit(2)

    backend = (args.backend or "st").lower()
    model = (
        args.model
        or (
            os.getenv("OPENAI_EMB") if backend == "openai" else os.getenv("EMBED_MODEL")
        )
        or (
            "text-embedding-3-small"
            if backend == "openai"
            else "sentence-transformers/allenai-specter"
        )
    )

    print(f"[build] EMBED_BACKEND={backend} | MODEL={model} | N={len(texts)}")
    if backend == "openai":
        emb = _embed_openai(texts, model)
        # OpenAI vectors are not unit-normalized; keep metric='ip' or normalize as needed.
    elif backend == "st":
        emb = _embed_st(texts, model)
    else:
        print(f"[build] Unknown backend: {backend}", file=sys.stderr)
        sys.exit(3)

    dim = emb.shape[1]
    index = _build_faiss(emb, args.metric, args.out)
    meta = {
        "ids": ids,
        "text_key": args.text_key,
        "dim": dim,
        "metric": args.metric,
        "backend": backend,
        "model": model,
        "count": len(ids),
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[build] Wrote FAISS index → {args.out} (ntotal={index.ntotal}, dim={dim}, metric={args.metric})"
    )
    print(f"[build] Wrote meta        → {meta_out}")


if __name__ == "__main__":
    main()
