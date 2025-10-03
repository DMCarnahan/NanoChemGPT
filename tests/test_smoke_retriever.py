from retriever import retriever
from pathlib import Path


def test_env_paths_and_safe_float(monkeypatch, tmp_path):
    # Ensure _env_paths returns a dict and _safe_float handles bad input
    monkeypatch.delenv('RETRIEVER_INDEX_DIRS', raising=False)
    monkeypatch.delenv('RETRIEVER_INDEX_DIR_DOC', raising=False)
    monkeypatch.delenv('RETRIEVER_INDEX_DIR_PASSAGE', raising=False)

    # Create index_doc directory to exercise heuristic branch
    base = Path(retriever.__file__).parent
    idx = base / 'index_doc'
    idx.mkdir(parents=True, exist_ok=True)

    env = retriever._env_paths()
    assert isinstance(env, dict)

    assert retriever._safe_float('not-a-number', 0.5) == 0.5
