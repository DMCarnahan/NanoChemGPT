import os
from importlib import reload
from pathlib import Path
import retriever.retriever as retr


def _clear_state():
    retr._BUNDLES.clear()
    retr._VECS.clear()
    # reset warning flags
    if hasattr(retr, '_MISSING_BUNDLE_WARNED'):
        retr._MISSING_BUNDLE_WARNED = False
    if hasattr(retr, '_AUTO_BUILD_ATTEMPTED'):
        retr._AUTO_BUILD_ATTEMPTED.clear()


def test_resolver_env_primary(tmp_path, monkeypatch):
    bundle = tmp_path / 'primary.jsonl'
    bundle.write_text('{}\n', encoding='utf-8')
    monkeypatch.setenv('BUNDLE_AUTO', str(bundle))
    reload(retr)  # re-import to pick new env for constants fallback
    resolved = retr._resolve_bundle_path()
    assert resolved == bundle.resolve()


def test_resolver_fallback(tmp_path, monkeypatch):
    # No BUNDLE_AUTO; create HARVEST_OUT_DIR with bundle -> should pick env harvest bundle
    harvest = tmp_path / 'harvest_out'
    harvest.mkdir()
    b = harvest / 'bundle.jsonl'
    b.write_text('{}\n', encoding='utf-8')
    monkeypatch.delenv('BUNDLE_AUTO', raising=False)
    monkeypatch.setenv('HARVEST_OUT_DIR', str(harvest))
    reload(retr)
    resolved = retr._resolve_bundle_path()
    assert resolved == b.resolve()


def test_resolver_missing(tmp_path, monkeypatch):
    # Point BUNDLE_AUTO to a non-existent temp file and unset harvest dir
    missing = tmp_path / 'no' / 'such' / 'bundle.jsonl'
    monkeypatch.setenv('BUNDLE_AUTO', str(missing))
    monkeypatch.delenv('HARVEST_OUT_DIR', raising=False)
    reload(retr)
    resolved = retr._resolve_bundle_path()
    assert resolved == missing.resolve()
    assert not resolved.exists()

