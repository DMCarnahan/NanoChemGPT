from pathlib import Path
import os

# Lightweight, side-effect-free defaults for common directories and flags.
# These are intentionally conservative and can be overridden via environment
# variables in production or CI.
ROOT = Path(__file__).resolve().parents[1]

ATTACH_DIR = Path(os.getenv("ATTACH_DIR", str(ROOT / "data" / "attachments")))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(ROOT / "data" / "uploads")))
LOOKUP_UPLOAD_DIR = Path(os.getenv("LOOKUP_UPLOAD_DIR", str(UPLOADS_DIR / "lookup")))
BUILTIN_DIR = Path(os.getenv("BUILTIN_DIR", str(ROOT / "builtin")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(ROOT / "data" / "index")))

# BUNDLE_AUTO historically was used as either a boolean or path; prefer a
# path-like default here (out/bundle_auto.jsonl) but allow env override.
_ba = os.getenv("BUNDLE_AUTO") or os.getenv("BUNDLE_AUTO_PATH")
if _ba:
    BUNDLE_AUTO = Path(_ba)
else:
    BUNDLE_AUTO = Path(
        os.getenv("BUNDLE_AUTO_PATH", str(ROOT / "out" / "bundle_auto.jsonl"))
    )
