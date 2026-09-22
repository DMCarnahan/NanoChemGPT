import subprocess

import pytest


class _EmptyUploads:
    def search(self, *_args, **_kwargs):
        return []


class _Collection:
    def insert_one(self, *_args, **_kwargs):
        return None


class _Database:
    qa = _Collection()


@pytest.fixture
def ask_module(monkeypatch):
    import app

    monkeypatch.setitem(app.app.config, "TESTING", True)
    monkeypatch.delenv("OFFLINE_TESTS", raising=False)
    monkeypatch.setattr(
        app.UploadsVectorSearch,
        "from_folder",
        lambda *_args, **_kwargs: _EmptyUploads(),
    )
    monkeypatch.setattr(app, "_h_kb_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(app, "retriever_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(app, "get_db", lambda: _Database())
    return app


@pytest.mark.parametrize(
    ("auto_harvest", "include_allow_fetch", "expected_calls"),
    [
        ("1", True, 1),
        ("0", True, 0),
        ("1", False, 0),
    ],
)
def test_ask_never_harvests_inline(
    ask_module,
    monkeypatch,
    auto_harvest,
    include_allow_fetch,
    expected_calls,
):
    monkeypatch.setenv("ENABLE_AUTO_HARVEST", auto_harvest)
    calls = []

    def fake_enqueue(query, **kwargs):
        calls.append((query, kwargs))
        return "job_test_123"

    def fail_if_started(*_args, **_kwargs):
        raise AssertionError("request handler attempted to launch a subprocess")

    monkeypatch.setattr(ask_module, "enqueue_text_mining_job", fake_enqueue)
    monkeypatch.setattr(subprocess, "Popen", fail_if_started)

    payload = {"question": "How can I synthesize iron oxide nanorods?"}
    if include_allow_fetch:
        payload["allow_fetch"] = True

    response = ask_module.app.test_client().post("/ask", json=payload)

    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert len(calls) == expected_calls
    assert data["mining_enqueued"] is bool(expected_calls)
    assert data["mining_job_id"] == (
        "job_test_123" if expected_calls else None
    )


def test_openai_failure_returns_structured_502(ask_module, monkeypatch):
    monkeypatch.setitem(ask_module.app.config, "TESTING", False)
    monkeypatch.setattr(
        ask_module,
        "retriever_search",
        lambda *_args, **_kwargs: [
            {
                "text": "iron oxide nanorod synthesis " * 50,
                "score": 0.95,
                "meta": {
                    "title": "Iron oxide nanorod synthesis",
                    "doi": "10.1000/example",
                },
            }
        ],
    )

    class FailingCompletions:
        @staticmethod
        def create(**_kwargs):
            raise TimeoutError("upstream timed out")

    class FailingChat:
        completions = FailingCompletions()

    class FailingClient:
        chat = FailingChat()

    monkeypatch.setattr(ask_module, "client", FailingClient())

    response = ask_module.app.test_client().post(
        "/ask",
        json={
            "question": "Explain iron oxide nanorod synthesis.",
            "mode": "reasoning",
        },
    )

    assert response.status_code == 502
    assert response.get_json() == {
        "ok": False,
        "error": "Language model request failed",
        "error_type": "TimeoutError",
    }


def test_healthz_reports_answer_readiness(ask_module, monkeypatch, tmp_path):
    from retriever import retriever as retriever_runtime

    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "tfidf.npz").write_bytes(b"matrix")
    (index_path / "vectorizer.joblib").write_bytes(b"vectorizer")
    (index_path / "rows.jsonl").write_text(
        '{"text": "example", "title": "Example"}\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        retriever_runtime,
        "_labels_and_paths",
        lambda: [("doc", index_path)],
    )

    response = ask_module.app.test_client().get("/healthz")

    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert data["ready"] is True
    assert data["openai_configured"] is True
    assert data["retriever_ready"] is True
    assert data["indexes"]["doc"]["rows_present"] is True
