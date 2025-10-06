from pathlib import Path
import os

# Minimal, import-safe path defaults. Keep these simple to avoid runtime side
# effects during static analysis. Tests and CI can set env vars to override.
ROOT = Path(__file__).resolve().parents[1]

ATTACH_DIR = Path(os.getenv("ATTACH_DIR", str(ROOT / "data" / "attachments"))).resolve()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(ROOT / "data" / "uploads"))).resolve()
LOOKUP_UPLOAD_DIR = Path(os.getenv("LOOKUP_UPLOAD_DIR", str(UPLOADS_DIR / "lookup"))).resolve()
BUILTIN_DIR = Path(os.getenv("BUILTIN_DIR", str(ROOT / "builtin"))).resolve()

_env_ba = os.getenv("BUNDLE_AUTO", "1")
try:
    BUNDLE_AUTO = bool(int(_env_ba))
except Exception:
    BUNDLE_AUTO = True


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
