"""
Lightweight vector search over an uploads folder.

- Works with existing call:
    UploadsVectorSearch.from_folder(uploads_dir, device=vector_device)
- ALSO accepts extra kwargs like backend='tfidf' and max_docs via **kwargs (ignored if unknown).

Backends:
- SentenceTransformers (if installed) for semantic search.
- Pure-Python TF‑IDF fallback (memory light).

Results:
[{"title","path","score","snippet","meta":{...}}, ...]
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

# ---------------- optional ST backend ----------------
_ST = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _ST = SentenceTransformer
except Exception:
    _ST = None

# --- Sentence-Transformers loader that also works for plain HF models ---
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models as st_models
except Exception:
    SentenceTransformer = None
    st_models = None


def _st_available() -> bool:
    return SentenceTransformer is not None


def _load_st_model(model_name: str, device: str = "cpu"):
    """
    Try to load a Sentence-Transformers model; if that fails, build an ST pipeline
    from a plain HF checkpoint (e.g., pranav-s/MaterialsBERT, m3rg-iitd/matscibert).
    """
    if not _st_available():
        return None

    # Allow short names like "all-MiniLM-L6-v2"
    if "/" not in model_name and not model_name.startswith("sentence-transformers/"):
        model_name = f"sentence-transformers/{model_name}"

    # 1) Try as ready-made ST model
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        # 2) Fallback: build ST pipeline from plain HF
        if st_models is None:
            raise
        tr = st_models.Transformer(model_name, max_seq_length=512)
        pool = st_models.Pooling(
            tr.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        return SentenceTransformer(modules=[tr, pool], device=device)


# ---------------- simple text I/O ---------------------
_HTML_TAG_RX = re.compile(r"<[^>]+>")
_WS_RX = re.compile(r"\s+")
_TOKEN_RX = re.compile(r"[A-Za-z0-9_]+")


def _read_text(p: Path) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return p.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""


def _read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            return "\n".join(
                page.extract_text() or "" for page in PdfReader(str(p)).pages
            )
        except Exception:
            return ""


def _read_docx(p: Path) -> str:
    try:
        import docx  # type: ignore

        d = docx.Document(str(p))
        return "\n".join(par.text for par in d.paragraphs if par.text)
    except Exception:
        return ""


def _read_html(p: Path) -> str:
    raw = _read_text(p)
    if not raw:
        return ""
    cleaned = _HTML_TAG_RX.sub(" ", raw)
    return _WS_RX.sub(" ", cleaned).strip()


def _read_json(p: Path) -> str:
    try:
        return json.dumps(
            json.loads(_read_text(p) or "{}"), ensure_ascii=False, indent=2
        )
    except Exception:
        return _read_text(p)


def _extract_text(p: Path) -> str:
    suf = p.suffix.lower()
    if suf in (".txt", ".md", ".rst", ".log"):
        return _read_text(p)
    if suf == ".pdf":
        return _read_pdf(p)
    if suf == ".docx":
        return _read_docx(p)
    if suf in (".html", ".htm"):
        return _read_html(p)
    if suf == ".json":
        return _read_json(p)
    return _read_text(p)


def _tokenize(t: str) -> List[str]:
    return [w.lower() for w in _TOKEN_RX.findall(t)]


# ---------------- tiny TF‑IDF backend ----------------
class _MiniTfidf:
    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self.doc_vectors: List[List[float]] = []
        self.doc_norms: List[float] = []
        self.docs: List["_Doc"] = []

    @staticmethod
    def _tf(tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for w in tokens:
            tf[w] = tf.get(w, 0.0) + 1.0
        n = float(len(tokens) or 1.0)
        for w in tf:
            tf[w] /= n
        return tf

    def fit(self, docs: List["_Doc"]) -> None:
        self.docs = docs
        df: Dict[str, int] = {}
        for d in docs:
            for w in set(_tokenize(d.text)):
                df[w] = df.get(w, 0) + 1
        self.vocab = {
            w: i for i, (w, _) in enumerate(sorted(df.items(), key=lambda kv: kv[0]))
        }
        N = max(1, len(docs))
        self.idf = [0.0] * len(self.vocab)
        for w, i in self.vocab.items():
            self.idf[i] = math.log((N + 1.0) / (df[w] + 1.0)) + 1.0
        self.doc_vectors = []
        self.doc_norms = []
        for d in docs:
            tf = self._tf(_tokenize(d.text))
            vec = [0.0] * len(self.vocab)
            for w, t in tf.items():
                idx = self.vocab.get(w)
                if idx is not None:
                    vec[idx] = t * self.idf[idx]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            self.doc_vectors.append(vec)
            self.doc_norms.append(norm)

    def query(self, q: str, top_k: int = 5) -> List[Tuple[int, float]]:
        tf = self._tf(_tokenize(q))
        qv = [0.0] * len(self.vocab)
        for w, t in tf.items():
            idx = self.vocab.get(w)
            if idx is not None:
                qv[idx] = t * self.idf[idx]
        qn = math.sqrt(sum(v * v for v in qv)) or 1.0
        scores: List[Tuple[int, float]] = []
        for i, dv in enumerate(self.doc_vectors):
            dot = sum(a * b for a, b in zip(qv, dv))
            scores.append((i, float(dot / (qn * self.doc_norms[i]))))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------- ST backend (batched) ---------------
class _STBackend:
    def __init__(
        self, model_name: str = "allenai-specter", device: str = "cpu"
    ) -> None:
        m = _load_st_model(model_name, device=device)
        if m is None:
            raise RuntimeError(
                "sentence-transformers not installed. `pip install sentence-transformers`"
            )
        self.model = m
        self.docs: List["_Doc"] = []
        self.corpus_emb = None

        try:
            dim = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            dim = 768
        self._bs = 32 if dim >= 700 or "bert" in model_name.lower() else 128

    def fit(self, docs: List["_Doc"]) -> None:
        import numpy as np

        self.docs = docs
        texts = [d.text for d in docs]
        embs = []
        bs = self._bs
        for i in range(0, len(texts), bs):
            e = self.model.encode(
                texts[i : i + bs],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embs.append(np.asarray(e, dtype=np.float32))
        self.corpus_emb = np.vstack(embs) if embs else None

    def query(self, q: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.docs or self.corpus_emb is None:
            return []
        import numpy as np

        q_emb = self.model.encode(
            [q],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sims = self.corpus_emb @ q_emb
        idxs = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[int(i)])) for i in idxs]


# ---------------- data model -------------------------
@dataclass
class _Doc:
    path: Path
    text: str
    title: str
    meta: Dict[str, Union[str, int, float]]


# ---------------- public API -------------------------
class UploadsVectorSearch:
    def __init__(
        self, docs: List[_Doc], backend_obj: Union[_MiniTfidf, _STBackend]
    ) -> None:
        self._docs = docs
        self._backend = backend_obj
        self._lower = [d.text.lower() for d in docs]

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        device: str = "cpu",
        pattern: Sequence[str] = (
            "*.txt",
            "*.md",
            "*.pdf",
            "*.docx",
            "*.html",
            "*.htm",
            "*.json",
        ),
        max_bytes: int = 2_000_000,
        st_model: str = "all-MiniLM-L6-v2",
        **kwargs,
    ) -> "UploadsVectorSearch":
        """
        Accepts optional kwargs:
          - backend: 'auto'|'tfidf'|'st' (env UPLOADS_VECTOR_BACKEND also supported)
          - max_docs: int (default 2000)
        Unknown kwargs are ignored for compatibility.
        """
        backend = str(kwargs.get("backend", "auto")).lower()
        max_docs = int(kwargs.get("max_docs", 2000))
        env_choice = os.getenv("UPLOADS_VECTOR_BACKEND", "").strip().lower()
        if env_choice in ("tfidf", "st"):
            backend = env_choice or backend

        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"uploads folder not found: {folder}")

        # collect files
        paths: List[Path] = []
        for patt in pattern:
            paths.extend(folder.rglob(patt))
        docs: List[_Doc] = []
        count = 0
        for p in sorted(set(paths)):
            try:
                if p.is_dir():
                    continue
                size = p.stat().st_size
                if size <= 0 or size > max_bytes:
                    continue
                text = _extract_text(p)
                if not text.strip():
                    continue
                if len(text) > 100_000:
                    text = text[:100_000]
                docs.append(
                    _Doc(
                        path=p.resolve(),
                        text=text,
                        title=p.name,
                        meta={"size": int(size), "relpath": str(p.relative_to(folder))},
                    )
                )
                count += 1
                if count >= max_docs:
                    break
            except Exception:
                continue

        # choose backend safely
        use_st = _st_available() and (
            backend == "st" or (backend == "auto" and len(docs) <= max_docs // 2)
        )
        if use_st:
            try:
                be: Union[_MiniTfidf, _STBackend] = _STBackend(
                    model_name=st_model, device=device
                )
                be.fit(docs)  # type: ignore
            except MemoryError:
                be = _MiniTfidf()
                be.fit(docs)  # type: ignore
            except Exception:
                be = _MiniTfidf()
                be.fit(docs)  # type: ignore
        else:
            be = _MiniTfidf()
            be.fit(docs)  # type: ignore

        return cls(docs, be)

    def __len__(self) -> int:
        return len(self._docs)

    def _snippet(self, i: int, query: str, radius: int = 160) -> str:
        t = self._lower[i]
        pos = 0
        for tok in _tokenize(query):
            j = t.find(tok)
            if j != -1:
                pos = j
                break
        s = max(0, pos - radius)
        e = min(len(t), pos + radius)
        sn = self._docs[i].text[s:e].replace("\n", " ").strip()
        return _WS_RX.sub(" ", sn)

    def search(
        self, query: str, top_k: int = 5, **kwargs
    ) -> List[Dict[str, Union[str, float, dict]]]:
        if not query or not self._docs:
            return []
        # alias k -> top_k
        if "k" in kwargs and (top_k is None or top_k == 5):
            try:
                top_k = int(kwargs["k"])
            except Exception:
                pass
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = 5

        hits = self._backend.query(query, top_k=top_k)  # type: ignore
        out: List[Dict[str, Union[str, float, dict]]] = []
        for idx, score in hits:
            d = self._docs[idx]
            out.append(
                {
                    "title": d.title,
                    "path": str(d.path),
                    "score": float(score),
                    "snippet": self._snippet(idx, query),
                    "meta": d.meta,
                }
            )
        return out


# Compat helpers (optional)
def search(
    query: str, top_k: int = 5, folder: Union[str, Path] = "uploads"
) -> List[Dict[str, Union[str, float, dict]]]:
    return UploadsVectorSearch.from_folder(
        folder=folder, device="cpu", st_model="pranav-s/MaterialsBERT", backend="st"
    ).search(query, top_k=top_k)


def rebuild() -> bool:
    return True
