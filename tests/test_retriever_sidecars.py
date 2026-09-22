import json

import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def test_npz_loader_attaches_rows_sidecar(monkeypatch, tmp_path):
    from retriever import retriever

    rows = [
        {
            "text": "Hydrothermal growth of iron oxide nanorods.",
            "title": "Iron oxide nanorods",
            "doi": "10.1000/nanorods",
        },
        {
            "text": "Annealing controls the nanorod crystal phase.",
            "title": "Annealing study",
            "url": "https://example.test/annealing",
        },
    ]
    texts = [row["text"] for row in rows]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    save_npz(tmp_path / "tfidf.npz", matrix)
    joblib.dump(vectorizer, tmp_path / "vectorizer.joblib")
    (tmp_path / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    monkeypatch.setenv("RETRIEVER_PREFER_NPZ", "1")
    retriever.reload_caches()
    try:
        bundle = retriever._load_tfidf_for(tmp_path, force=True)
        assert bundle["texts"] == texts
        assert bundle["metas"] == rows
        assert any(meta.get("doi") for meta in bundle["metas"])
    finally:
        retriever.reload_caches()
