from __future__ import annotations
import os, tempfile
from pathlib import Path

# Root of repo (anchor)
ROOT = Path(__file__).resolve().parent.parent

# Base data dir (override with DATA_DIR)
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data")).resolve()


def _mk(p: Path) -> Path:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p



ATTACH_DIR = _mk(Path(os.getenv("ATTACH_DIR", DATA_DIR / "attachments")))
UPLOADS_DIR = _mk(Path(os.getenv("UPLOADS_DIR", DATA_DIR / "uploads")))
LOOKUP_UPLOAD_DIR = _mk(Path(os.getenv("LOOKUP_UPLOAD_DIR", DATA_DIR / "lookup_uploads")))
BUILTIN_DIR = _mk(Path(os.getenv("BUILTIN_DIR", ROOT / "builtin")))
HARVEST_OUT_DIR = _mk(Path(os.getenv("HARVEST_OUT_DIR", "/data/harvester/out_auto")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "/data/vector_store")).resolve()
RETRIEVER_INDEX_DIR_DOC = Path(os.getenv("RETRIEVER_INDEX_DIR_DOC", "/data/vector_store_doc")).resolve()
RETRIEVER_INDEX_DIR_PASSAGE = Path(os.getenv("RETRIEVER_INDEX_DIR_PASSAGE", "/data/vector_store_passage")).resolve()
BUNDLE_PATH = os.getenv("BUNDLE_PATH", str(HARVEST_OUT_DIR / "bundle.jsonl"))

# Auto bundle (harvester output JSONL)
BUNDLE_AUTO = os.getenv("BUNDLE_AUTO", str(HARVEST_OUT_DIR / "bundle.jsonl"))
LOOKUP_DUCKDB_PATH = os.getenv("LOOKUP_DUCKDB_PATH", "/data/reactions.duckdb")
LOOKUP_DUCKDB_TABLE = os.getenv("LOOKUP_DUCKDB_TABLE", "reactions")
LOOKUP_PARQUET_GLOB = os.getenv("LOOKUP_PARQUET_GLOB", "/data/*.parquet")
LOOKUP_TEXT_COLS = os.getenv("LOOKUP_TEXT_COLS", "procedure,solvent,notes,title")
KB_TEXTS_PATH = os.getenv("KB_TEXTS_PATH", "/data/vector_store/texts.jsonl")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/data/vector_store/index.faiss")
HARVEST_COOLDOWN_MIN = int(os.getenv("HARVEST_COOLDOWN_MIN", "60"))
JUDGE_MIN_CHARS = int(os.getenv("JUDGE_MIN_CHARS", "64"))
JUDGE_MIN_HITS = int(os.getenv("JUDGE_MIN_HITS", "1"))
JUDGE_MIN_SCORE = float(os.getenv("JUDGE_MIN_SCORE", "0.15"))
MINER_DELAY_SECONDS = int(os.getenv("MINER_DELAY_SECONDS", "0"))
MINER_MAX_JOBS = int(os.getenv("MINER_MAX_JOBS", "2"))
MINER_MODE = os.getenv("MINER_MODE", "redis")
USE_DB_CONTEXT = os.getenv("USE_DB_CONTEXT", "0") == "1"
VS_LLM_RERANK = os.getenv("VS_LLM_RERANK", "0") == "1"
VS_MMR = os.getenv("VS_MMR", "0") == "1"
PERSIST_INDEX = os.getenv("PERSIST_INDEX", "0") == "1"
ENABLE_AUTO_HARVEST = os.getenv("ENABLE_AUTO_HARVEST", "").lower() in {"1", "true", "yes", "on"}
ENABLE_ENHANCED_CITATIONS = os.getenv("ENABLE_ENHANCED_CITATIONS", "").lower() in {"1", "true", "yes", "on"}

# Fallback temp (used if primary is not writable)
def ensure_writable(p: Path) -> Path:
    try:
        test = p / ".perm_test"
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("x", encoding="utf-8")
        test.unlink(missing_ok=True)
        return p
    except Exception:
        tmp = Path(tempfile.gettempdir()) / "nanochem_fallback" / p.name
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

ATTACH_DIR = ensure_writable(ATTACH_DIR)
UPLOADS_DIR = ensure_writable(UPLOADS_DIR)
LOOKUP_UPLOAD_DIR = ensure_writable(LOOKUP_UPLOAD_DIR)
HARVEST_OUT_DIR = ensure_writable(HARVEST_OUT_DIR)

__all__ = [
    "ROOT",
    "DATA_DIR",
    "ATTACH_DIR",
    "UPLOADS_DIR",
    "LOOKUP_UPLOAD_DIR",
    "BUILTIN_DIR",
    "HARVEST_OUT_DIR",
    "INDEX_DIR",
    "RETRIEVER_INDEX_DIR_DOC",
    "RETRIEVER_INDEX_DIR_PASSAGE",
    "BUNDLE_PATH",
    "LOOKUP_DUCKDB_PATH",
    "LOOKUP_DUCKDB_TABLE",
    "LOOKUP_PARQUET_GLOB",
    "LOOKUP_TEXT_COLS",
    "KB_TEXTS_PATH",
    "FAISS_INDEX_PATH",
    "HARVEST_COOLDOWN_MIN",
    "JUDGE_MIN_CHARS",
    "JUDGE_MIN_HITS",
    "JUDGE_MIN_SCORE",
    "MINER_DELAY_SECONDS",
    "MINER_MAX_JOBS",
    "MINER_MODE",
    "USE_DB_CONTEXT",
    "VS_LLM_RERANK",
    "VS_MMR",
    "PERSIST_INDEX",
    "ENABLE_AUTO_HARVEST",
    "ENABLE_ENHANCED_CITATIONS",
    "BUNDLE_AUTO",
]
