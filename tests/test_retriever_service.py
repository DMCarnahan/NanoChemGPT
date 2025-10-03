import json
from pathlib import Path

import pytest

# Import the FastAPI service module directly and call endpoint functions to avoid
# depending on FastAPI's TestClient in environments where fastapi isn't installed.
import retriever.service as svc

def test_health_empty_index(monkeypatch, tmp_path):
    # Ensure _state is clean
    svc._state["index"] = None
    svc._state["texts"] = []

    j = svc.health()
    assert j["ok"] is True
    assert j["ntotal"] == 0
    assert j["texts"] == 0

def test_search_no_index(monkeypatch):
    svc._state["index"] = None
    svc._state["texts"] = []

    j = svc.search(svc.SearchIn(q="cobalt oxide"))
    assert isinstance(j.get("hits"), list)
    assert len(j["hits"]) == 0

def test_search_with_index(monkeypatch):
    # Create a fake index-like object with a simple search API
    class FakeIndex:
        def __init__(self, ntotal, vectors):
            self._ntotal = ntotal
            self.vectors = vectors
        @property
        def ntotal(self):
            return self._ntotal
        def search(self, qv, k):
            # always return top-1 index 0 with score 0.9
            import numpy as np
            D = np.array([[0.9]])
            I = np.array([[0]])
            return D, I

    # Monkeypatch embed to return a deterministic vector
    monkeypatch.setattr('retriever.service.embed', lambda texts: [[0.1] * 1536])
    fake_texts = [{"text": "Example doc about cobalt oxide", "id": 0}]
    svc._state["index"] = FakeIndex(1, None)
    svc._state["texts"] = fake_texts

    j = svc.search(svc.SearchIn(q="cobalt oxide", k=1))
    assert "hits" in j
    assert isinstance(j["hits"], list)
    assert len(j["hits"]) == 1
    hit = j["hits"][0]
    assert hit["i"] == 0
    assert abs(hit["score"] - 0.9) < 1e-6