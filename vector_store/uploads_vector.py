"""
Lightweight vector search over your uploads folder.

Provides:
    - class UploadsVectorSearch
        .from_folder(path, device='cpu', pattern=('*.txt','*.md','*.pdf','*.docx','*.html','*.htm','*.json'))
        .search(query: str, top_k: int = 5) -> list[dict]
        .rebuild() -> None
        .add_paths(paths: list[str|Path]) -> None
        .__len__()

Design:
    - Tries to use SentenceTransformers if available for semantic search.
    - Otherwise falls back to a tiny in-house TF‑IDF implementation (no deps).
    - Extracts text from .txt/.md/.pdf/.docx/.html/.json (best-effort).
    - Returns records: {title, path, score, snippet, meta}

This is intentionally dependency-light and safe: if extractors fail, files are skipped.
"""

from __future__ import annotations
import io
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# -------------------------- Optional backends --------------------------
_ST = None
try:
    # Prefer 'sentence_transformers'
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST = SentenceTransformer
except Exception:
    _ST = None

@dataclass
class _Doc:
    path: Path
    text: str
    title: str
    meta: Dict[str, Union[str, int, float]]

# ----------------------- Simple text extractors ------------------------

_HTML_TAG_RX = re.compile(r"<[^>]+>")
_WS_RX = re.compile(r"\s+")

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""

def _read_pdf(path: Path) -> str:
    # Try pypdf first, then PyPDF2
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return ""

def _read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs if p.text)
    except Exception:
        return ""

def _read_html(path: Path) -> str:
    raw = _read_text(path)
    if not raw:
        return ""
    # Strip tags, collapse whitespace
    cleaned = _HTML_TAG_RX.sub(" ", raw)
    return _WS_RX.sub(" ", cleaned).strip()

def _read_json(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return _read_text(path)

def _extract_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".txt", ".md", ".rst", ".log"):
        return _read_text(path)
    if suf == ".pdf":
        return _read_pdf(path)
    if suf == ".docx":
        return _read_docx(path)
    if suf in (".html", ".htm"):
        return _read_html(path)
    if suf == ".json":
        return _read_json(path)
    # Try best-effort text read
    return _read_text(path)

# ------------------------ Tiny TF‑IDF backend --------------------------

_TOKEN_RX = re.compile(r"[A-Za-z0-9_]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RX.findall(text)]

class _MiniTfidf:
    """
    Minimal TF‑IDF that needs only Python + math.
    Uses dense lists for simplicity; fine for a few thousand docs.
    """
    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self.doc_vectors: List[List[float]] = []
        self.doc_norms: List[float] = []
        self.docs: List[_Doc] = []

    @staticmethod
    def _tf(tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for w in tokens:
            tf[w] = tf.get(w, 0.0) + 1.0
        n = float(len(tokens) or 1.0)
        for w in tf:
            tf[w] /= n
        return tf

    def fit(self, docs: List[_Doc]) -> None:
        self.docs = docs
        # Build DF
        df: Dict[str, int] = {}
        for d in docs:
            uniq = set(_tokenize(d.text))
            for w in uniq:
                df[w] = df.get(w, 0) + 1
        # Vocab
        self.vocab = {w: i for i, (w, _) in enumerate(sorted(df.items(), key=lambda kv: kv[0]))}
        N = max(1, len(docs))
        # IDF
        self.idf = [0.0] * len(self.vocab)
        for w, i in self.vocab.items():
            self.idf[i] = math.log((N + 1.0) / (df[w] + 1.0)) + 1.0
        # Doc vectors
        self.doc_vectors = []
        self.doc_norms = []
        for d in docs:
            tf = self._tf(_tokenize(d.text))
            vec = [0.0] * len(self.vocab)
            for w, t in tf.items():
                idx = self.vocab.get(w)
                if idx is not None:
                    vec[idx] = t * self.idf[idx]
            norm = math.sqrt(sum(v*v for v in vec)) or 1.0
            self.doc_vectors.append(vec)
            self.doc_norms.append(norm)

    def query(self, q: str, top_k: int = 5) -> List[Tuple[int, float]]:
        qtf = self._tf(_tokenize(q))
        qvec = [0.0] * len(self.vocab)
        for w, t in qtf.items():
            idx = self.vocab.get(w)
            if idx is not None:
                qvec[idx] = t * self.idf[idx]
        qnorm = math.sqrt(sum(v*v for v in qvec)) or 1.0
        # Cosine similarity
        scores: List[Tuple[int, float]] = []
        for i, dvec in enumerate(self.doc_vectors):
            dot = sum(a*b for a, b in zip(qvec, dvec))
            sim = dot / (qnorm * self.doc_norms[i])
            scores.append((i, float(sim)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# -------------------- SentenceTransformers backend ---------------------

class _STBackend:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self.model = _ST(model_name, device=device)  # type: ignore
        self.corpus_emb = None  # type: ignore
        self.docs: List[_Doc] = []

    def fit(self, docs: List[_Doc]) -> None:
        self.docs = docs
        texts = [d.text for d in docs]
        self.corpus_emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def query(self, q: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.docs:
            return []
        q_emb = self.model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        # cosine similarity == dot product due to normalization
        import numpy as np  # type: ignore
        sims = (self.corpus_emb @ q_emb)  # type: ignore
        idxs = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[int(i)])) for i in idxs]

# -------------------------- Public interface ---------------------------

class UploadsVectorSearch:
    def __init__(self, docs: List[_Doc], backend: Union[_MiniTfidf, _STBackend]) -> None:
        self._docs = docs
        self._backend = backend
        # Precompute for snippets
        self._lower_texts = [d.text.lower() for d in docs]

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        device: str = "cpu",
        pattern: Sequence[str] = ("*.txt", "*.md", "*.pdf", "*.docx", "*.html", "*.htm", "*.json"),
        max_bytes: int = 2_000_000,  # 2 MB per file
        st_model: str = "all-MiniLM-L6-v2"
    ) -> "UploadsVectorSearch":
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"uploads folder not found: {folder}")
        # Gather files
        paths: List[Path] = []
        for patt in pattern:
            paths.extend(folder.rglob(patt))
        # Build docs
        docs: List[_Doc] = []
        for p in sorted(set(paths)):
            try:
                if p.is_dir():
                    continue
                size = p.stat().st_size
                if size <= 0 or size > max_bytes:
                    continue
                text = _extract_text(p)
                if not text or not text.strip():
                    continue
                # Soft trim to 100k chars to keep things lean
                if len(text) > 100_000:
                    text = text[:100_000]
                docs.append(_Doc(
                    path=p.resolve(),
                    text=text,
                    title=p.name,
                    meta={"size": int(size), "relpath": str(p.relative_to(folder))}
                ))
            except Exception:
                # skip unreadable files
                continue

        # Choose backend
        if _ST is not None:
            backend: Union[_MiniTfidf, _STBackend] = _STBackend(model_name=st_model, device=device)
        else:
            backend = _MiniTfidf()

        backend.fit(docs)  # type: ignore
        return cls(docs, backend)

    def __len__(self) -> int:
        return len(self._docs)

    def rebuild(self) -> None:
        # No-op for now (recreate instance via from_folder)
        return None

    def add_paths(self, paths: Iterable[Union[str, Path]]) -> None:
        # Simple incremental add: append then refit backend
        new_docs: List[_Doc] = []
        for p in paths:
            P = Path(p)
            if not P.exists() or P.is_dir():
                continue
            try:
                size = P.stat().st_size
                text = _extract_text(P)
                if not text:
                    continue
                new_docs.append(_Doc(
                    path=P.resolve(),
                    text=text[:100_000],
                    title=P.name,
                    meta={"size": int(size), "relpath": P.name}
                ))
            except Exception:
                continue
        if not new_docs:
            return
        docs = self._docs + new_docs
        # Rebuild backend with all docs
        if isinstance(self._backend, _STBackend):
            self._backend.fit(docs)
        else:
            self._backend.fit(docs)  # _MiniTfidf has same API
        self._docs = docs
        self._lower_texts = [d.text.lower() for d in docs]

    def _make_snippet(self, i: int, query: str, radius: int = 160) -> str:
        doc_lower = self._lower_texts[i]
        q_tokens = [t for t in _tokenize(query) if t]
        pos = -1
        for qt in q_tokens:
            pos = doc_lower.find(qt)
            if pos != -1:
                break
        if pos == -1:
            pos = 0
        start = max(0, pos - radius)
        end = min(len(doc_lower), pos + radius)
        snippet = self._docs[i].text[start:end].replace("\n", " ").strip()
        # Collapse whitespace
        snippet = _WS_RX.sub(" ", snippet)
        return snippet

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float, dict]]]:
        if not query or not self._docs:
            return []
        hits: List[Tuple[int, float]]
        hits = self._backend.query(query, top_k=top_k)  # type: ignore
        results: List[Dict[str, Union[str, float, dict]]] = []
        for idx, score in hits:
            d = self._docs[idx]
            results.append({
                "title": d.title,
                "path": str(d.path),
                "score": float(score),
                "snippet": self._make_snippet(idx, query),
                "meta": d.meta,
            })
        return results

# Backwards-compat simple functions (optional)
def search(query: str, top_k: int = 5, folder: Union[str, Path] = "uploads") -> List[Dict[str, Union[str, float, dict]]]:
    vs = UploadsVectorSearch.from_folder(folder)
    return vs.search(query, top_k=top_k)

def rebuild() -> bool:
    return True
