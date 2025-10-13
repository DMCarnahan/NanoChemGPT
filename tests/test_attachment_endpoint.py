import os
from pathlib import Path
from app import app as flask_app
from app_utils.constants import ATTACH_DIR


def test_attachment_text_endpoint(tmp_path, monkeypatch):
    # Create a fake text attachment
    ATTACH_DIR.mkdir(parents=True, exist_ok=True)
    aid = "testabc12345"
    (ATTACH_DIR / f"{aid}.txt").write_text(
        "Example attachment text content.", encoding="utf-8"
    )

    client = flask_app.test_client()
    resp = client.get(f"/attachment_text/{aid}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["meta"]["id"] == aid
    assert data["snippet"].startswith("Example attachment")


def test_attachment_text_pdf_fail_cache(tmp_path, monkeypatch):
    # Simulate a PDF that previously failed extraction by creating .fail marker
    ATTACH_DIR.mkdir(parents=True, exist_ok=True)
    aid = "pdffail123456"
    # create original pdf file placeholder
    pdf_path = ATTACH_DIR / f"{aid}__sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
    fail_marker = ATTACH_DIR / f"{aid}.fail"
    fail_marker.write_text("empty", encoding="utf-8")

    from app_utils.uploads import read_attachment_text

    txt = read_attachment_text(aid)
    # Cached fail should return empty string
    assert txt == ""
