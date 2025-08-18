import argparse, json, pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_bundle(bundle_path: str) -> list[dict]:
    out = []
    with open(bundle_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def extract_docs(bundle: list[dict]) -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    for rec in bundle:
        pid = rec.get("paper_id") or rec.get("doi") or rec.get("title")
        title = rec.get("title","")
        url = rec.get("urls",{}).get("pdf")
        license = rec.get("license")
        for para in rec.get("extractions",{}).get("methods_paragraphs",[]):
            t = para.get("text","").strip()
            if not t:
                continue
            meta = {
                "paper_id": pid,
                "title": title,
                "url": url,
                "license": license,
                "ents": para.get("ents",[]),
                "links": para.get("links") or para.get("ops") or []
            }
            texts.append(t)
            metas.append(meta)
    return texts, metas

# -------- Embedding backends --------

def embed_openai(texts: list[str], model: str) -> np.ndarray:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. pip install openai>=1.30.0") from e
    client = OpenAI()
    # Chunk to stay under token limits; simple batching
    B = 128
    embs = []
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=model, input=batch)
        embs.extend([d.embedding for d in resp.data])
    return np.array(embs, dtype="float32")

def embed_sentencetransformers(texts: list[str], model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers") from e
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embs, dtype="float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--embed-backend", choices=["openai","sentence-transformers", "none"], default="none")
    ap.add_argument("--embed-model", type=str, default=None, help="e.g., text-embedding-3-large or all-MiniLM-L6-v2")
    args = ap.parse_args()

    Path(args.index_dir).mkdir(parents=True, exist_ok=True)

    # Load & prep
    bundle = load_bundle(args.bundle)
    texts, metas = extract_docs(bundle)
    if args.max_docs:
        texts, metas = texts[:args.max_docs], metas[:args.max_docs]

    # ----- TF-IDF -----
    vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2)
    X = vec.fit_transform(texts)
    with open(Path(args.index_dir)/"tfidf.pkl", "wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": X, "metas": metas, "texts": texts}, f)
    print(f"[tfidf] Indexed {len(texts)} paragraphs into {args.index_dir}")

    # ----- Embeddings (optional) -----
    if args.embed-backend != "none":
        model = args.embed_model or ("text-embedding-3-large" if args.embed-backend=="openai" else "all-MiniLM-L6-v2")
        if args.embed-backend == "openai":
            E = embed_openai(texts, model)
        else:
            E = embed_sentencetransformers(texts, model)
        # L2-normalize
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
        E = E / norms
        with open(Path(args.index_dir)/"embed.pkl", "wb") as f:
            pickle.dump({"backend": args.embed-backend, "model": model, "embeddings": E, "metas": metas, "texts": texts}, f)
        print(f"[embed] Built embeddings with {args.embed-backend}:{model} → {E.shape}")

if __name__ == "__main__":
    main()
