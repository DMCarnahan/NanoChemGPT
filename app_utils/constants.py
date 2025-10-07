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

# Canonical writable dirs (env override -> fallback -> temp)
ATTACH_DIR = _mk(Path(os.getenv("ATTACH_DIR", DATA_DIR / "attachments")))
UPLOADS_DIR = _mk(Path(os.getenv("UPLOADS_DIR", DATA_DIR / "uploads")))
LOOKUP_UPLOAD_DIR = _mk(Path(os.getenv("LOOKUP_UPLOAD_DIR", DATA_DIR / "lookup_uploads")))
BUILTIN_DIR = _mk(Path(os.getenv("BUILTIN_DIR", ROOT / "builtin")))
HARVEST_OUT_DIR = _mk(Path(os.getenv("HARVEST_OUT_DIR", "harvester/out_auto")))
INDEX_DIR = Path(os.getenv("RETRIEVER_INDEX_DIR_DOC", "retriever/index_doc")).resolve()

# Auto bundle (harvester output JSONL)
BUNDLE_AUTO = os.getenv("BUNDLE_AUTO") or os.getenv("BUNDLE_AUTO_PATH") or str(HARVEST_OUT_DIR / "bundle.jsonl")

# Feature flags
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
    "BUNDLE_AUTO",
    "ENABLE_AUTO_HARVEST",
    "ENABLE_ENHANCED_CITATIONS",
]
