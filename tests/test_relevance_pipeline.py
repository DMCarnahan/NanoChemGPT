from types import SimpleNamespace


def test_nanorod_query_relevance_and_references(monkeypatch):
    import app

    # 1) Fake retriever returns a relevant hit
    def fake_retriever(query, k=8, level=None, **kwargs):
        return [
            {
                "meta": {
                    "doi": "10.1000/sn2020",
                    "title": "Synthesis of SnO nanorods with controlled aspect ratio",
                    "authors": ["Alice"],
                },
                "text": "We synthesized SnO nanorods by hydrothermal method...",
                "score": 0.95,
            }
        ]

    monkeypatch.setattr(app, "retriever_search", fake_retriever)

    # 2) Fake harvest returns additional references
    def fake_harvest(queries, use_grobid=None, jid=None):
        return [
            {
                "title": "Hydrothermal growth of SnO nanorods",
                "year": "2019",
                "doi": "10.2000/sn_hydro",
                "authors": ["Bob"],
            }
        ]

    # _harvest_reindex is defined inside the ask handler; insert our fake at module level
    monkeypatch.setattr(app, "_harvest_reindex", fake_harvest, raising=False)

    # Exercise the live request path through a deterministic Responses API stub.
    class FakeClient:
        class responses:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(
                    output_text="You can synthesize SnO nanorods via hydrothermal methods [1].",
                    id="resp_relevance",
                    model="test-model",
                    usage={},
                )

    monkeypatch.setattr(app, "client", FakeClient())
    monkeypatch.setitem(app.app.config, "TESTING", False)
    monkeypatch.delenv("OFFLINE_TESTS", raising=False)

    # 4) Disable auto-harvest so the test doesn't spawn subprocesses in CI/test env
    monkeypatch.setenv("ENABLE_AUTO_HARVEST", "0")

    client = app.app.test_client()
    resp = client.post(
        "/ask",
        json={
            "question": "how can i synthesize diameter 10 nm length 50 nm SnO nanorods?",
            "allow_fetch": True,
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("ok") is True

    # The server should return a 'refs' block assembled from retriever + harvest
    refs = data.get("refs") or []
    titles = [r.get("title", "").lower() for r in refs]
    assert any("nanorod" in t or "sno" in t or "sn" in t for t in titles), (
        f"Unexpected refs: {titles}"
    )
