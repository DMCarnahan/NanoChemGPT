from app import app as flask_app
import re

def test_verbatim_normalization(monkeypatch):
    client = flask_app.test_client()

    dotted = "·m·a·g·n·e·t·i·c· ·p·r·o·p·e·r·t·i·e·s· were studied."  # tail normal words to ensure not fully dropped

    # Monkeypatch helpers used in /ask path to force verbatim mode
    import app as app_module

    def fake_wants_verbatim(q: str):
        return True  # Force verbatim

    def fake_read_attachment_text(aid: str, max_pages: int = 40):
        return dotted

    def fake_best_chunks_from_text(txt, qtext, top_k=3):
        return [txt]

    monkeypatch.setattr(app_module, '_wants_verbatim', fake_wants_verbatim)
    monkeypatch.setattr(app_module, 'read_attachment_text', fake_read_attachment_text)
    monkeypatch.setattr(app_module, '_best_chunks_from_text', fake_best_chunks_from_text)

    # Issue request with attachment id to trigger attachment context build
    resp = client.post('/ask', json={'question': 'verbatim procedure', 'attachment_ids': ['ATT1']})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('ok') is True
    answer = data.get('answer') or ''
    # Ensure the dotted pattern collapsed to readable word 'magnetic'
    assert 'magnetic' in answer.lower(), f"Normalized word missing. Answer=\n{answer}"
    # Ensure original excessive dot pattern not present
    assert '·m·a·g·n·e·t·i·c' not in answer
