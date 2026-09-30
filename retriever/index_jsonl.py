from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

MIN_CHARS_DEFAULT = 40
INDEX_SCHEMA_VERSION = 2

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

    # URL: prefer explicit PDF/URL, include a 'urls':{'pdf':...}
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

    # Authors: prefer list from meta; fall back to legacy helper fields
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


def build_vectorizer() -> FeatureUnion:
    """Build a lexical index that handles both prose and chemical notation."""

    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b[\w-]{2,}\b",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    max_features=160_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "chem_char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=90_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_tfidf_for_jsonl(
    bundle: str | Path,
    index_dir: str | Path,
    *,
    text_key: str = "methods",
    min_chars: int = MIN_CHARS_DEFAULT,
    max_docs: int | None = None,
) -> dict:
    """Build an index for CLI, startup preflight, and background rebuilds."""

    bundle_path = Path(bundle).resolve()
    out = Path(index_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ids, texts, metas = load_texts(bundle_path, text_key, min_chars)
    if max_docs:
        ids, texts, metas = (
            ids[:max_docs],
            texts[:max_docs],
            metas[:max_docs],
        )

    if not texts:
        raise ValueError(
            f"No documents to index from {bundle_path} (source='{text_key}')."
        )

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)
    if X.shape[1] == 0:
        raise ValueError("No index features; inputs are likely too short or empty.")

    from scipy.sparse import save_npz

    npz_final = out / "tfidf.npz"
    tmp_npz = out / "tfidf.tmp.npz"

    save_npz(tmp_npz, X)
    os.replace(tmp_npz, npz_final)

    vectorizer_final = out / "vectorizer.joblib"
    vectorizer_tmp = out / "vectorizer.tmp.joblib"
    joblib.dump(vectorizer, vectorizer_tmp)
    os.replace(vectorizer_tmp, vectorizer_final)
    if os.getenv("WRITE_LEGACY_TFIDF_PKL", "0").lower() in {"1", "true", "yes"}:
        joblib.dump(
            {
                "matrix": X,
                "vectorizer": vectorizer,
                "texts": texts,
                "metas": metas,
            },
            out / "tfidf.pkl",
        )
    rows_final = out / "rows.jsonl"
    rows_tmp = out / "rows.tmp.jsonl"
    with rows_tmp.open("w", encoding="utf-8") as f:
        for t, m in zip(texts, metas):
            row = {"text": t}
            row.update(m)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(rows_tmp, rows_final)

    metadata = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "documents": len(ids),
        "features": int(X.shape[1]),
        "text_key": text_key,
        "vectorizer": "word_and_chemical_character_tfidf",
    }
    metadata_final = out / "index_meta.json"
    metadata_tmp = out / "index_meta.tmp.json"
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_tmp, metadata_final)
    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--index_dir", required=True, type=Path)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument(
        "--embed-backend",
        choices=["openai", "sentence-transformers", "none"],
        default="none",
    )  # compatibility; this builder is deliberately local and deterministic
    ap.add_argument("--embed-model", default=None)  # compatibility
    ap.add_argument(
        "--text-key",
        default="methods",
        help="methods | sections | raw | text | content | abstract | body | title",
    )
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS_DEFAULT)
    args = ap.parse_args()

    metadata = build_tfidf_for_jsonl(
        args.bundle,
        args.index_dir,
        text_key=args.text_key,
        min_chars=args.min_chars,
        max_docs=args.max_docs,
    )

    print(
        "[index_jsonl] OK. "
        f"docs={metadata['documents']} terms={metadata['features']} "
        f"→ {args.index_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
