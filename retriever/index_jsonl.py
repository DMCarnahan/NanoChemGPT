from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Iterable, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from joblib import dump

MIN_CHARS_DEFAULT = 40

def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec

def _pick_id(rec: Dict, i: int) -> str:
    for k in ("paper_id","id","uid","doc_id","hash","doi"):
        v = rec.get(k)
        if v:
            return str(v)
    return f"rec:{i}"

def _normalize_text(s: str | None) -> str:
    return " ".join((s or "").split())

def _gather_text(rec: Dict, source: str) -> str:
    """
    source options:
      - 'methods'   → join extractions.methods_paragraphs[*].text
      - 'sections'  → join sections[*].text
      - any field name → top-level str (with fallbacks)
    """
    if source == "methods":
        mps = (rec.get("extractions") or {}).get("methods_paragraphs") or []
        paras = []
        for it in mps:
            if isinstance(it, dict):
                t = it.get("text")
                if isinstance(t, str) and t.strip():
                    paras.append(t.strip())
        return "\n\n".join(paras)

    if source == "sections":
        secs = rec.get("sections") or []
        paras = []
        for s in secs:
            if isinstance(s, dict):
                t = s.get("text")
                if isinstance(t, str) and t.strip():
                    paras.append(t.strip())
        return "\n\n".join(paras)

    # top-level field or sensible fallbacks
    t = rec.get(source)
    if not isinstance(t, str):
        for k in ("raw","text","content","abstract","body","title"):
            tv = rec.get(k)
            if isinstance(tv, str) and tv.strip():
                t = tv; break
        else:
            t = ""
    return t

def load_texts(path: Path, source: str, min_chars: int) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    for i, rec in enumerate(_iter_jsonl(path)):
        t = _normalize_text(_gather_text(rec, source))
        if len(t) < min_chars:
            # try a gentle fallback: if methods empty, try sections; else meta
            if source == "methods":
                t = _normalize_text(_gather_text(rec, "sections"))
            if len(t) < min_chars:
                meta = []
                for k in ("title","abstract"):
                    v = rec.get(k)
                    if isinstance(v, str) and v.strip():
                        meta.append(v.strip())
                t = _normalize_text("\n\n".join(meta))
        if len(t) < min_chars:
            continue
        ids.append(_pick_id(rec, i))
        texts.append(t)
    return ids, texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--index_dir", required=True, type=Path)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--embed-backend", choices=["openai","sentence-transformers","none"], default="none")  # compat
    ap.add_argument("--embed-model", default=None)  # compat
    ap.add_argument("--text-key", default="methods", help="methods | sections | raw | text | content | abstract | body | title")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS_DEFAULT)
    args = ap.parse_args()

    # Create output dir first
    out = args.index_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Load texts
    ids, texts = load_texts(args.bundle, args.text_key, args.min_chars)
    if args.max_docs:
        ids, texts = ids[:args.max_docs], texts[:args.max_docs]

    if not texts:
        raise SystemExit(f"[index_jsonl] No documents to index from {args.bundle} (source='{args.text_key}').")

    # Vectorize
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[\w-]{2,}\b",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.99,
        max_features=250_000,
    )
    X = vectorizer.fit_transform(texts)
    if X.shape[1] == 0:
        raise SystemExit("[index_jsonl] 0 features — inputs likely too short/empty.")

    # Minimal metas 
    metas = [{"id": i} for i in ids]

    from scipy.sparse import save_npz
    import joblib, json, os

    npz_final = out / "tfidf.npz"
    tmp_npz   = out / "tfidf.npz.tmp"     

    try:
        save_npz(tmp_npz, X)              # write tmp
        os.replace(tmp_npz, npz_final)    # atomic move if tmp exists
    except FileNotFoundError:
        save_npz(npz_final, X)

    joblib.dump(vectorizer, out / "vectorizer.joblib")
    joblib.dump({"matrix": X, "vectorizer": vectorizer, "texts": texts,
                "metas": [{"id": i} for i in ids]}, out / "tfidf.pkl")

if __name__ == "__main__":
    main()
