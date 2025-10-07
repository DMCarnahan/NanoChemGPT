import os
import sys
from pathlib import Path
import pytest

from retriever.retriever import _ensure_tfidf_index

class DummyBuilder:
    built = False

# Monkeypatch build function indirectly by creating a dummy file situation

def test_index_permission_fallback(monkeypatch, tmp_path):
    # Simulate permission error on first mkdir
    calls = {"n": 0}

    class FakePath(Path):
        _flavour = Path('.')._flavour  # required by pathlib subclass

        def mkdir(self, parents=False, exist_ok=False):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError("no write")
            return super().mkdir(parents=parents, exist_ok=exist_ok)

    target = tmp_path / "deny_dir"
    fake = FakePath(str(target))

    # Patch Path in scope of _ensure_tfidf_index -> easiest is to patch idx_path passed in
    # Also patch _resolve_bundle_path to return an existing empty bundle
    bundle = tmp_path / "bundle.jsonl"
    bundle.write_text("{}\n")

    from retriever import retriever as R

    monkeypatch.setattr(R, "_resolve_bundle_path", lambda: bundle)

    # Patch build_tfidf_for_jsonl import path by creating a stub module function
    def fake_build(bundle_path, out_dir):
        (Path(out_dir) / "tfidf.pkl").write_text("ok")

    monkeypatch.setitem(sys.modules, 'retriever.index_jsonl', type('X', (), {'build_tfidf_for_jsonl': fake_build}))

    _ensure_tfidf_index(fake)
    # After fallback, a tfidf.pkl should exist either in a fallback dir under /tmp/nanochem_indexes
    fallback_root = Path('/tmp/nanochem_indexes')
    assert fallback_root.exists()
    found = list(fallback_root.rglob('tfidf.pkl'))
    assert found, 'Expected tfidf.pkl in fallback root'
