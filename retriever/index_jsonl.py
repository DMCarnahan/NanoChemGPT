from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

MIN_CHARS_DEFAULT = 40

_DOI_RX = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)


def _norm_doi_any(x):
    if not x:
        return ""
    m = _DOI_RX.search(str(x))
    return m.group(1).lower() if m else ""


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
    for k in ("paper_id", "id", "uid", "doc_id", "hash", "doi"):
        v = rec.get(k)
        if v:
            return str(v)
    return f"rec:{i}"


def _author_names(auths) -> List[str]:
    out: List[str] = []
    if isinstance(auths, list):
        for a in auths:
            if isinstance(a, str):
                out.append(a)
            elif isinstance(a, dict):
                n = (
                    a.get("name")
                    or " ".join(x for x in [a.get("first"), a.get("last")] if x)
                    or " ".join(x for x in [a.get("given"), a.get("family")] if x)
                )
                if n:
                    out.append(n)
    return out


def _pick_meta(rec: Dict) -> Dict:
    # prefer the harvester's rich block
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}

    title = meta.get("title") or rec.get("title") or rec.get("name") or ""

    # DOI: extract from multiple candidates
    doi = _norm_doi_any(
        meta.get("doi")
        or rec.get("doi")
        or rec.get("paper_id")
        or rec.get("url")
        or rec.get("oa_url")
    )

    # URL: prefer explicit PDF/URL, include your 'urls':{'pdf':...}
    url = (
        meta.get("pdf_url")
        or meta.get("url")
        or (rec.get("urls") or {}).get("pdf")
        or rec.get("url")
        or rec.get("oa_url")
        or rec.get("pdf_url")
        or ""
    )

    # Year
    year = meta.get("year") or rec.get("year") or rec.get("publication_year")
    if not year:
        for k in ("date", "published", "pub_date"):
            v = rec.get(k)
            if isinstance(v, str) and len(v) >= 4 and v[:4].isdigit():
                year = v[:4]
                break

    # Authors: prefer list from meta; fall back to your helper/legacy fields
    authors = meta.get("authors")
    if not authors:
        authors = _author_names(rec.get("authors") or rec.get("authorships") or []) or (
            rec.get("authors") or []
        )

    return {
        "title": title,
        "doi": doi,
        "url": url,
        "year": str(year or ""),
        "authors": authors,
    }


def _normalize_text(s: str | None) -> str:
    return " ".join((s or "").split())


def _gather_text(rec: Dict, source: str) -> str:
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
    t = rec.get(source)
    if not isinstance(t, str):
        for k in ("raw", "text", "content", "abstract", "body", "title"):
            tv = rec.get(k)
            if isinstance(tv, str) and tv.strip():
                t = tv
                break
        else:
            t = ""
    return t


def load_texts(
    path: Path, source: str, min_chars: int
) -> Tuple[List[str], List[str], List[Dict]]:
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict] = []
    for i, rec in enumerate(_iter_jsonl(path)):
        t = _normalize_text(_gather_text(rec, source))
        if len(t) < min_chars:
            if source == "methods":
                t = _normalize_text(_gather_text(rec, "sections"))
            if len(t) < min_chars:
                meta = []
                for k in ("title", "abstract"):
                    v = rec.get(k)
                    if isinstance(v, str) and v.strip():
                        meta.append(v.strip())
                t = _normalize_text("\n\n".join(meta))
        if len(t) < min_chars:
            continue
        rid = _pick_id(rec, i)
        ids.append(rid)
        texts.append(t)
        m = _pick_meta(rec)
        m["id"] = rid
        metas.append(m)
    return ids, texts, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--index_dir", required=True, type=Path)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument(
        "--embed-backend",
        choices=["openai", "sentence-transformers", "none"],
        default="none",
    )  # compat
    ap.add_argument("--embed-model", default=None)  # compat
    ap.add_argument(
        "--text-key",
        default="methods",
        help="methods | sections | raw | text | content | abstract | body | title",
    )
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS_DEFAULT)
    args = ap.parse_args()

    out = args.index_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ids, texts, metas = load_texts(args.bundle, args.text_key, args.min_chars)
    if args.max_docs:
        ids, texts, metas = (
            ids[: args.max_docs],
            texts[: args.max_docs],
            metas[: args.max_docs],
        )

    if not texts:
        raise SystemExit(
            f"[index_jsonl] No documents to index from {args.bundle} (source='{args.text_key}')."
        )

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

    import os

    import joblib
    from scipy.sparse import save_npz

    npz_final = out / "tfidf.npz"
    tmp_npz = out / "tfidf.npz.tmp"

    try:
        save_npz(tmp_npz, X)
        os.replace(tmp_npz, npz_final)
    except FileNotFoundError:
        save_npz(npz_final, X)

    joblib.dump(vectorizer, out / "vectorizer.joblib")
    joblib.dump(
        {"matrix": X, "vectorizer": vectorizer, "texts": texts, "metas": metas},
        out / "tfidf.pkl",
    )
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for t, m in zip(texts, metas):
            row = {"text": t}
            row.update(m)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[index_jsonl] OK. docs={len(ids)} terms={X.shape[1]} → {out}")


if __name__ == "__main__":
    main()
