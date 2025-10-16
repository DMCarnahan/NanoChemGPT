from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


TFIDF_FILES = [
    "tfidf.pkl",
    "tfidf.npz",
    "vectorizer.joblib",
    "rows.jsonl",
]


def _truthy(val: str | None) -> bool:
    return (val or "").lower() in {"1", "true", "yes", "on"}


def purge_index_dir(idx: Path) -> None:
    idx.mkdir(parents=True, exist_ok=True)
    for name in TFIDF_FILES:
        try:
            (idx / name).unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def build_index(bundle: Path, idx: Path, text_key: str, min_chars: int = 40) -> int:
    idx.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "retriever" / "index_jsonl.py"),
        "--bundle",
        str(bundle),
        "--index_dir",
        str(idx),
        "--text-key",
        text_key,
        "--min-chars",
        str(min_chars),
    ]
    return subprocess.call(cmd)


def main() -> int:
    # Env-driven config
    rebuild = _truthy(os.getenv("REBUILD_TFIDF"))
    if not rebuild:
        return 0

    # Inputs
    harvest_dir = Path(os.getenv("HARVEST_OUT_DIR", "/data/harvester/out_auto")).resolve()
    bundle_methods = Path(os.getenv("BUNDLE_PATH", harvest_dir / "bundle_with_methods.jsonl")).resolve()
    bundle_plain = Path(harvest_dir / "bundle.jsonl").resolve()

    # Index dirs
    idx_single = Path(os.getenv("RETRIEVER_INDEX_DIR", os.getenv("INDEX_DIR", "/data/vector_store"))).resolve()
    idx_doc = Path(os.getenv("RETRIEVER_INDEX_DIR_DOC", "/data/vector_store_doc")).resolve()
    idx_pas = Path(os.getenv("RETRIEVER_INDEX_DIR_PASSAGE", "/data/vector_store_passage")).resolve()

    # Optional purge
    if _truthy(os.getenv("PURGE_TFIDF")):
        for p in {idx_single, idx_doc, idx_pas}:
            purge_index_dir(p)

    rc_total = 0

    # Build passage (methods) if bundle exists, else fall back
    if bundle_methods.exists() and bundle_methods.stat().st_size > 0:
        rc = build_index(bundle_methods, idx_pas, text_key="methods", min_chars=20)
        rc_total |= rc
        # Also single-index if set
        if str(idx_single) != str(idx_pas):
            rc = build_index(bundle_methods, idx_single, text_key="methods", min_chars=20)
            rc_total |= rc
    elif bundle_plain.exists() and bundle_plain.stat().st_size > 0:
        # Fallback to sections/abstract
        rc = build_index(bundle_plain, idx_pas, text_key="sections", min_chars=20)
        if rc != 0:
            rc = build_index(bundle_plain, idx_pas, text_key="abstract", min_chars=10)
        rc_total |= rc
        if str(idx_single) != str(idx_pas):
            rc = build_index(bundle_plain, idx_single, text_key="sections", min_chars=20)
            if rc != 0:
                rc = build_index(bundle_plain, idx_single, text_key="abstract", min_chars=10)
            rc_total |= rc
    else:
        # Nothing to build
        return 0

    # Doc-level (abstract)
    if bundle_plain.exists() and bundle_plain.stat().st_size > 0:
        rc = build_index(bundle_plain, idx_doc, text_key="abstract", min_chars=10)
        rc_total |= rc

    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
