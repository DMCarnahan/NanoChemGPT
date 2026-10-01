import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from app_utils.attachment_context import build_attachment_context


def test_complete_protocol_is_preserved_with_its_final_workup():
    protocol = (
        Path(__file__).parent / "fixtures" / "ctab_screen_protocol.txt"
    ).read_text()
    context, metadata = build_attachment_context(
        [("protocol1", protocol)], "Optimize wire aspect ratio", maximum_chars=30_000
    )
    assert protocol.strip() in context
    assert metadata == [
        {
            "id": "protocol1",
            "source_chars": len(protocol.strip()),
            "context_chars": len(protocol.strip()),
            "excerpted": False,
        }
    ]


def test_small_protocol_stays_complete_alongside_a_long_paper():
    paper = "Background discussion of nanostructures. " * 2_000
    protocol = "Add 10 mL water. Stir at 500 rpm for 5 minutes."
    context, metadata = build_attachment_context(
        [("paper", paper), ("protocol", protocol)],
        "Optimize synthesis",
        maximum_chars=4_000,
    )
    assert len(context) <= 4_000
    assert protocol in context
    assert metadata[0]["excerpted"] is True
    assert metadata[1]["excerpted"] is False
    assert "Attachment excerpted by server" in context


def test_long_document_excerpts_keep_source_order_and_relevant_methods():
    paper = (
        ("Unrelated background. " * 1_000)
        + "\n\nPtNi synthesis: Add 10 mL water and 5 mL PtNi stock.\n\n"
        + "Stir the PtNi synthesis mixture at 500 rpm for 5 minutes.\n\n"
        + ("Unrelated discussion. " * 1_000)
    )
    context, metadata = build_attachment_context(
        [("paper", paper)], "PtNi synthesis", maximum_chars=4_000
    )
    assert len(context) <= 4_000
    assert context.index("Add 10 mL") < context.index("Stir the PtNi")
    assert metadata[0]["excerpted"] is True


@pytest.fixture
def attachment_app(monkeypatch, tmp_path):
    import app
    import app_utils.uploads as uploads

    requests = []

    class Responses:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                output_text=(
                    "## Synthesis Protocol:\n3. **Procedure**\n[\n"
                    "1. Define the target:\n   Aspect ratio 6.\n"
                    "2. Add 10 mL water to the flask.\n"
                    "3. Stir at 500 rpm for 5 minutes.\n]\n"
                ),
                id="resp_attachment",
                model="test-model",
                usage={},
            )

    monkeypatch.setitem(app.app.config, "TESTING", False)
    monkeypatch.delenv("OFFLINE_TESTS", raising=False)
    monkeypatch.setenv("ENABLE_AUTO_HARVEST", "0")
    monkeypatch.setattr(app, "client", SimpleNamespace(responses=Responses()))
    monkeypatch.setattr(app, "_h_kb_search", lambda *_a, **_kw: [])
    monkeypatch.setattr(app, "retriever_search", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        app,
        "get_db",
        lambda: SimpleNamespace(qa=SimpleNamespace(insert_one=lambda *_a, **_kw: None)),
    )
    monkeypatch.setattr(uploads, "ATTACH_DIR", tmp_path)
    return app, requests


def test_uploaded_protocol_reaches_model_input_and_converter(attachment_app):
    app, requests = attachment_app
    source = (
        Path(__file__).parent / "fixtures" / "ctab_screen_protocol.txt"
    ).read_text()
    client = app.app.test_client()
    uploaded = client.post(
        "/attach",
        data={"files": (io.BytesIO(source.encode()), "protocol.txt")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 200
    aid = uploaded.get_json()["items"][0]["id"]
    response = client.post(
        "/ask",
        json={
            "question": "Optimize the attached synthesis for aspect ratio 6.",
            "attachments": [aid],
            "robot_ops": True,
        },
    )
    assert response.status_code == 200
    assert len(requests) == 1
    assert source.strip() in requests[0]["input"]
    assert f"[A1.1] attachment:{aid}" in requests[0]["input"]
    data = response.get_json()
    assert data["attachments_used"][0]["excerpted"] is False
    assert data["robot_operations"]["micro_plan"]
    assert data["executor_valid"] is True
    assert "generated_pddl" not in data["robot_operations"]


@pytest.mark.parametrize("text", ["", "   ", None])
def test_unreadable_attachment_stops_before_model_request(
    attachment_app, monkeypatch, text
):
    app, requests = attachment_app
    monkeypatch.setattr(app, "read_attachment_text", lambda *_a, **_kw: text)
    response = app.app.test_client().post(
        "/ask",
        json={
            "question": "Optimize the attached protocol.",
            "attachments": ["missing"],
        },
    )
    assert response.status_code == 422
    assert response.get_json()["error_code"] == "attachment_unreadable"
    assert response.get_json()["attachment_ids"] == ["missing"]
    assert requests == []


def test_failed_attachment_is_not_hidden_by_another_readable_file(
    attachment_app, monkeypatch
):
    app, requests = attachment_app
    monkeypatch.setattr(
        app,
        "read_attachment_text",
        lambda aid, **_kw: "Add 10 mL water." if aid == "good" else "",
    )
    response = app.app.test_client().post(
        "/ask", json={"question": "Use both protocols.", "attachments": ["good", "bad"]}
    )
    assert response.status_code == 422
    assert response.get_json()["attachment_ids"] == ["bad"]
    assert requests == []


def test_attachment_read_exception_returns_a_clear_error(attachment_app, monkeypatch):
    app, requests = attachment_app

    def fail(*_args, **_kwargs):
        raise OSError("cannot read attachment")

    monkeypatch.setattr(app, "read_attachment_text", fail)
    response = app.app.test_client().post(
        "/ask", json={"question": "Use this protocol.", "attachments": ["unreadable"]}
    )
    assert response.status_code == 422
    assert response.get_json()["error_code"] == "attachment_unreadable"
    assert requests == []


def test_retrieved_literature_cannot_clip_the_end_of_a_complete_attachment(
    attachment_app, monkeypatch
):
    app, requests = attachment_app
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "4000")
    aid = "near_limit"
    capacity = 4000 - len(f"<<<CTX_ATTACH>>>\n[A1.1] attachment:{aid}\n")
    tail = "Final workup: centrifuge at 9000 rpm for 5 minutes."
    source = "Synthesis notes. " * (capacity // 17)
    source = source[: capacity - len(tail)] + tail
    monkeypatch.setattr(app, "read_attachment_text", lambda *_a, **_kw: source)
    monkeypatch.setattr(
        app,
        "retriever_search",
        lambda *_a, **_kw: [
            {
                "text": "PtNi synthesis literature " * 100,
                "score": 0.95,
                "meta": {"title": "PtNi synthesis", "doi": "10.1000/example"},
            }
        ],
    )
    response = app.app.test_client().post(
        "/ask", json={"question": "Analyze the PtNi synthesis.", "attachments": [aid]}
    )
    assert response.status_code == 200
    assert source in requests[0]["input"]
    assert response.get_json()["attachments_used"][0]["excerpted"] is False


def test_uploaded_pdf_text_reaches_model_input(attachment_app):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    app, requests = attachment_app
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 72 700 Td (PtNi protocol: Add 10 mL water to the flask.) Tj "
        b"0 -20 Td (Stir at 500 rpm for 5 minutes.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    pdf = io.BytesIO()
    writer.write(pdf)
    pdf.seek(0)
    client = app.app.test_client()
    uploaded = client.post(
        "/attach",
        data={"files": (pdf, "protocol.pdf")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 200
    item = uploaded.get_json()["items"][0]
    assert item["n_chars"] > 0
    response = client.post(
        "/ask",
        json={"question": "Optimize the PtNi protocol.", "attachments": [item["id"]]},
    )
    assert response.status_code == 200
    assert "Add 10 mL water to the flask." in requests[0]["input"]
    assert "Stir at 500 rpm for 5 minutes." in requests[0]["input"]
    assert response.get_json()["attachments_used"][0]["excerpted"] is False
