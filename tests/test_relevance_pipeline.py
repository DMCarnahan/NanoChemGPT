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

    # 3) Fake OpenAI client that returns content citing [1]
    class FakeResp:
        def __init__(self, content):
            class Msg:
                def __init__(self, c):
                    self.content = c

            # provide both .choices and .choices[0].message.content style used elsewhere
            self.choices = [type("C", (), {"message": Msg(content)})()]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return FakeResp(
                        "You can synthesize SnO nanorods via hydrothermal methods [1].\n\n## References\n[1] Synthesis of SnO nanorods with controlled aspect ratio (2018)"
                    )

    monkeypatch.setattr(app, "client", FakeClient())

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
    assert any(
        "nanorod" in t or "sno" in t or "sn" in t for t in titles
    ), f"Unexpected refs: {titles}"
