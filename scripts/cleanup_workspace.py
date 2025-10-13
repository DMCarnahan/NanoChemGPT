"""Utility script to purge non-source build/test artifacts.

Removes common transient folders so the repo stays lean:
  - htmlcov/ (coverage HTML)
  - build/ (wheel build intermediates)
  - *.egg-info/ (setuptools metadata)
  - __pycache__/ (Python bytecode caches)
  - .pytest_cache/ (pytest cache)
  - .ruff_cache/ (ruff linter cache)

Keeps:
  - dist/ (release artifacts) unless --purge-dist specified.

Usage:
  python scripts/cleanup_workspace.py            # standard cleanup
  python scripts/cleanup_workspace.py --purge-dist  # also remove dist/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGET_DIRS = [
    "htmlcov",
    "build",
    "__pycache__",
    "NanoChemGPT.egg-info",
]
CACHE_DIRS = [".pytest_cache", ".ruff_cache"]


def remove_dir(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
        print(f"Removed {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--purge-dist", action="store_true", help="Also remove dist/ directory"
    )
    args = ap.parse_args()

    for name in TARGET_DIRS + CACHE_DIRS:
        remove_dir(ROOT / name)

    # remove nested __pycache__ recursively
    for pycache in ROOT.rglob("__pycache__"):
        remove_dir(pycache)

    if args.purge_dist:
        remove_dir(ROOT / "dist")
    else:
        if (ROOT / "dist").exists():
            print("Kept dist/ (release artifacts). Use --purge-dist to remove.")


if __name__ == "__main__":
    main()
