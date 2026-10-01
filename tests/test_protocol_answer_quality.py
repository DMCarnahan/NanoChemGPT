from types import SimpleNamespace

import pytest

from app_utils.citations import preserves_non_citation_text
from app_utils.evidence_relevance import filter_relevant_hits, material_scope
from app_utils.protocol_format import protocol_tables_to_lists


# Invented screening values test formatting without reproducing an attachment.
TABLE = (
    "| Batch | Stock concentration | Stock amount | Status |\n"
    "|---|---:|---:|---|\n"
    "| Control | 5 mM | 0.025 mmol | Attached condition [A1.1] |\n"
    "| Low stock | 2 mM | 0.010 mmol | **Proposed starting point** |\n"
    "| High stock | 8 mM | 0.040 mmol | **Proposed starting point**, only if homogeneous |"
)


def hit(title, text="Synthesis and growth experiments."):
    return {
        "meta": {"title": title, "doi": "10.1000/" + title[:5]},
        "text": text,
        "score": 0.99,
    }


def test_screening_table_becomes_labeled_bullets_without_losing_values():
    converted = protocol_tables_to_lists(TABLE)
    assert "|" not in converted
    assert (
        "- Batch: Control; Stock concentration: 5 mM; Stock amount: 0.025 mmol; Status: Attached condition [A1.1]"
        in converted
    )
    assert "2 mM" in converted and "0.010 mmol" in converted
    assert "8 mM" in converted and "only if homogeneous" in converted
    assert protocol_tables_to_lists(converted) == converted


def test_table_normalization_preserves_code_and_unrecognized_text():
    source = "```txt\n" + TABLE + "\n```\nPlain a | b."
    assert protocol_tables_to_lists(source) == source
    malformed = "| A | B |\n|---|---|\n| 10 mL | water | extra |"
    assert "| 10 mL | water | extra |" in protocol_tables_to_lists(malformed)


@pytest.mark.parametrize(
    "title",
    [
        "Application of nanotechnology in fruit crops-from synthesis to sustainable packaging",
        "Study on the Electro-Optical Properties of Polymer-Dispersed Liquid Crystals Doped with Cellulose Nanocrystals",
        "Autonomous multi-robot synthesis and optimization of metal halide perovskite nanocrystals",
        "Silver-decorated laser-induced graphene for a piezoresistive strain sensor",
        "Advanced Nanomedicines for Treating Refractory Inflammation-Related Diseases",
        "Tin oxide SnO2 nanorods with CTAB",
    ],
)
def test_high_retrieval_scores_do_not_qualify_other_materials(title):
    assert (
        filter_relevant_hits(
            [hit(title)], "Optimize PtNi nanowires for aspect ratio 6", ["PtNi"]
        )
        == []
    )


@pytest.mark.parametrize("name", ["PtNi", "Pt-Ni", "Pt/Ni", "platinum-nickel"])
def test_target_material_aliases_are_retained(name):
    source = hit(f"Growth of {name} nanowires")
    assert filter_relevant_hits([source], "PtNi nanowires", ["PtNi"]) == [source]


def test_attachment_can_supply_target_for_an_elliptical_question():
    scope = material_scope(
        "Optimize the attached protocol for aspect ratio 6",
        "PtNi nanowire synthesis: H2PtCl6, NiCl2, and NaBH4.",
    )
    assert scope == ["PtNi"]
    assert material_scope("Prepare Au nanoparticles", "PtNi nanowires") == ["Au"]
    assert material_scope("ptni nanowires") == ["PtNi"]
    assert material_scope("Pt/Ni nanowires") == ["PtNi"]
    assert material_scope("platinum-nickel nanowires") == ["PtNi"]
    assert material_scope("platinum and nickel nanowires") == ["PtNi"]


def test_scope_distinguishes_chemical_formulas_and_does_not_use_doi_or_author():
    assert material_scope("SnO nanorods at 20 C with 10 mL water") == ["SnO"]
    source = hit("Tin oxide SnO2 nanorods")
    assert filter_relevant_hits([source], "SnO nanorods", ["SnO"]) == []
    source["meta"]["doi"] = "10.1000/PtNi"
    source["meta"]["authors"] = ["PtNi"]
    assert filter_relevant_hits([source], "PtNi nanowires", ["PtNi"]) == []


def test_citation_pass_accepts_citations_but_rejects_changed_conditions():
    draft = (
        "Add 10 mL water.\n\n```reason\nReported growth uses this solvent [A1.1].\n```"
    )
    fixed = draft.replace("solvent [A1.1].", "solvent [1].")
    assert preserves_non_citation_text(draft, fixed)
    assert not preserves_non_citation_text(draft, fixed.replace("10 mL", "20 mL"))


@pytest.fixture
def quality_app(monkeypatch):
    import app

    calls = []
    queries = []
    draft = (
        "## Synthesis Protocol:\n2. **Materials**:\n[\n"
        + TABLE
        + "\n]\n3. **Procedure**\n[\n1. Add 10 mL water to the flask.\n]\n\n```reason\nReported PtNi growth uses water.\n```"
    )

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            text = (
                draft
                if len(calls) == 1
                else calls[-1]["input"]
                .split("BEGIN DRAFT\n", 1)[1]
                .split("\nEND DRAFT", 1)[0]
                .replace(
                    "Reported PtNi growth uses water.",
                    "Reported PtNi growth uses water. [1]",
                )
            )
            return SimpleNamespace(
                output_text=text, id="resp_quality", model="test-model", usage={}
            )

    def retrieve(query, **_kwargs):
        queries.append(query)
        return [
            hit("Fruit crops and nanotechnology"),
            hit("PtNi nanowires", "Reported PtNi growth uses water."),
        ]

    monkeypatch.setitem(app.app.config, "TESTING", False)
    monkeypatch.delenv("OFFLINE_TESTS", raising=False)
    monkeypatch.setenv("ENABLE_AUTO_HARVEST", "0")
    monkeypatch.setattr(app, "client", SimpleNamespace(responses=Responses()))
    monkeypatch.setattr(app, "_h_kb_search", lambda *_a, **_kw: [])
    monkeypatch.setattr(app, "retriever_search", retrieve)
    monkeypatch.setattr(
        app,
        "read_attachment_text",
        lambda *_a, **_kw: "PtNi nanowire synthesis: Add 10 mL water.",
    )
    monkeypatch.setattr(
        app,
        "get_db",
        lambda: SimpleNamespace(qa=SimpleNamespace(insert_one=lambda *_a, **_kw: None)),
    )
    return app, calls, queries


def test_model_and_export_use_lists_and_relevant_evidence_and_repair_rationale(
    quality_app,
):
    app, calls, queries = quality_app
    response = app.app.test_client().post(
        "/ask",
        json={
            "question": "Optimize the attached synthesis for aspect ratio 6",
            "attachments": ["protocol"],
            "robot_ops": True,
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "PtNi" in queries[0]
    assert "Fruit crops" not in calls[0]["input"]
    assert "PtNi nanowires" in calls[0]["input"]
    assert "|" not in data["answer"]
    assert "Reported PtNi growth uses water." in calls[1]["input"]
    assert "[1]" in data["rationale"]
    assert data["refs_used"][0]["title"] == "PtNi nanowires"
    assert len(data["refs_all"]) == 1
    assert data["grounding"]["rejected_hits"] == 1
    assert data["grounding"]["literature_evidence_sources"] == 1
    assert data["robot_operations"]["micro_plan"][0]["volume"] == 10


def test_no_relevant_literature_is_explicit_and_does_not_trigger_citation_padding(
    quality_app, monkeypatch
):
    app, calls, _queries = quality_app
    monkeypatch.setattr(
        app,
        "retriever_search",
        lambda *_a, **_kw: [hit("Fruit crops and nanotechnology")],
    )
    response = app.app.test_client().post(
        "/ask", json={"question": "PtNi nanowires", "attachments": ["protocol"]}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert len(calls) == 1
    assert data["refs_all"] == [] and data["refs_used"] == []
    assert data["grounding"]["literature_evidence_sources"] == 0
    assert data["grounding"]["rejected_hits"] == 1
