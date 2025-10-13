"""NanoChemGPT admin/ops helper CLI.

Usage:
  python scripts/nanochem_admin.py status
  python scripts/nanochem_admin.py build-tfidf --bundle harvester/out_auto/bundle.jsonl --out retriever/index_doc
  python scripts/nanochem_admin.py build-faiss --bundle harvester/out_auto/bundle.jsonl --index retriever/index_doc/index.faiss

This script is intentionally lightweight (no external deps). It provides a few
common maintenance actions for Railway or local ops.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def status():
    from app_utils.constants import HARVEST_OUT_DIR, INDEX_DIR

    bundle = Path(HARVEST_OUT_DIR) / "bundle.jsonl"
    tfidf_any = any((Path(INDEX_DIR) / n).exists() for n in ("tfidf.pkl", "tfidf.npz"))
    faiss_idx = Path(INDEX_DIR) / "index.faiss"
    info = {
        "bundle": str(bundle),
        "bundle_exists": bundle.exists(),
        "bundle_size": bundle.stat().st_size if bundle.exists() else 0,
        "tfidf_index": tfidf_any,
        "faiss_index": faiss_idx.exists(),
        "index_dir": str(INDEX_DIR),
    }
    print(json.dumps(info, indent=2))


def build_tfidf(bundle: str, out: str):
    from retriever.index_jsonl import build_tfidf_for_jsonl

    b = Path(bundle)
    if not b.exists():
        print(f"Bundle not found: {b}", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_tfidf_for_jsonl(str(b), str(out_dir))
    print(f"TF-IDF index built at {out_dir}")


def build_faiss(bundle: str, index_path: str):
    try:
        import faiss  # type: ignore
    except Exception:
        print("faiss not available", file=sys.stderr)
        sys.exit(3)
    from vector_store import embed

    b = Path(bundle)
    if not b.exists():
        print(f"Bundle not found: {b}", file=sys.stderr)
        sys.exit(2)
    docs = []
    with b.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except Exception:
                    continue
    texts = [d.get("text") or d.get("content") or "" for d in docs]
    texts = [t for t in texts if t]
    vecs = embed(texts)
    import numpy as np
    import faiss as _faiss

    X = np.asarray(vecs, dtype="float32")
    _faiss.normalize_L2(X)
    idx = _faiss.IndexFlatIP(X.shape[1])
    idx.add(X)
    outp = Path(index_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    _faiss.write_index(idx, str(outp))
    print(f"FAISS index written: {outp} (ntotal={idx.ntotal})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("status")
    p1 = sub.add_parser("build-tfidf")
    p1.add_argument("--bundle", required=True)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("build-faiss")
    p2.add_argument("--bundle", required=True)
    p2.add_argument("--index", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "status":
        status()
        return
    if args.cmd == "build-tfidf":
        build_tfidf(args.bundle, args.out)
        return
    if args.cmd == "build-faiss":
        build_faiss(args.bundle, args.index)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
