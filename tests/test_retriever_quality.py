import json

from retriever.index_jsonl import INDEX_SCHEMA_VERSION, build_tfidf_for_jsonl


def _write_bundle(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_hybrid_index_is_atomic_and_search_preserves_absolute_scores(
    monkeypatch, tmp_path
):
    from retriever import retriever

    bundle = tmp_path / "bundle.jsonl"
    index_dir = tmp_path / "index_doc"
    _write_bundle(
        bundle,
        [
            {
                "id": "iron-rods",
                "title": "Iron oxide nanorods",
                "doi": "10.1000/iron-rods",
                "abstract": (
                    "Hydrothermal Fe3O4 nanorod growth uses iron chloride and "
                    "controls anisotropic morphology through temperature and ligands."
                ),
            },
            {
                "id": "gold-seeds",
                "title": "Gold nanocrystal seeds",
                "doi": "10.1000/gold-seeds",
                "abstract": (
                    "Seed-mediated Au nanocrystal synthesis uses HAuCl4, citrate, "
                    "and controlled reduction to tune particle size."
                ),
            },
        ],
    )

    metadata = build_tfidf_for_jsonl(bundle, index_dir, text_key="abstract")

    assert metadata["schema_version"] == INDEX_SCHEMA_VERSION
    assert metadata["documents"] == 2
    assert (index_dir / "tfidf.npz").is_file()
    assert (index_dir / "vectorizer.joblib").is_file()
    assert (index_dir / "rows.jsonl").is_file()
    assert (index_dir / "index_meta.json").is_file()
    assert not (index_dir / "tfidf.tmp.npz").exists()
    assert not (index_dir / "rows.tmp.jsonl").exists()
    assert not (index_dir / "vectorizer.tmp.joblib").exists()
    assert not (index_dir / "index_meta.tmp.json").exists()
    assert not (index_dir / "tfidf.pkl").exists()

    monkeypatch.setenv("RETRIEVER_INDEX_DIR_DOC", str(index_dir))
    monkeypatch.delenv("RETRIEVER_INDEX_DIRS", raising=False)
    monkeypatch.delenv("RETRIEVER_INDEX_DIR_PASSAGE", raising=False)
    retriever.reload_caches()
    try:
        related = retriever.search("Fe3O4 iron oxide nanorods", k=1, w_doc=1.0)
        unrelated = retriever.search("banana orchard accounting", k=1, w_doc=1.0)
    finally:
        retriever.reload_caches()

    assert related["hits"][0]["meta"]["doi"] == "10.1000/iron-rods"
    assert related["hits"][0]["raw_score"] > 0.2
    # A query with no lexical or chemical overlap must not be rescaled to 1.0.
    assert unrelated["hits"][0]["raw_score"] < 0.05


def test_search_suppresses_duplicate_source_within_level(monkeypatch, tmp_path):
    from retriever import retriever

    bundle = tmp_path / "bundle.jsonl"
    index_dir = tmp_path / "index_doc"
    _write_bundle(
        bundle,
        [
            {
                "title": "Duplicate synthesis",
                "doi": "10.1000/duplicate",
                "abstract": "Cobalt oxide nanoparticle synthesis by thermal decomposition with oleic acid.",
            },
            {
                "title": "Duplicate synthesis copy",
                "doi": "10.1000/duplicate",
                "abstract": "Cobalt oxide nanoparticle synthesis using thermal decomposition and oleic acid.",
            },
        ],
    )
    build_tfidf_for_jsonl(bundle, index_dir, text_key="abstract")

    monkeypatch.setenv("RETRIEVER_INDEX_DIR_DOC", str(index_dir))
    monkeypatch.delenv("RETRIEVER_INDEX_DIRS", raising=False)
    monkeypatch.delenv("RETRIEVER_INDEX_DIR_PASSAGE", raising=False)
    retriever.reload_caches()
    try:
        result = retriever.search("cobalt oxide synthesis", k=5, w_doc=1.0)
    finally:
        retriever.reload_caches()

    assert len(result["hits"]) == 1
    assert result["hits"][0]["meta"]["doi"] == "10.1000/duplicate"
