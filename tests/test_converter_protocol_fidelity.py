from pathlib import Path

import pytest

from converter import (
    apply_postprocessing,
    convert_text_to_gt_schema,
    convert_text_to_robot_ops,
    extract_steps,
    validate_execution_plan,
)

PROTOCOL = (Path(__file__).parent / "fixtures" / "ctab_screen_protocol.txt").read_text()


@pytest.fixture
def screen():
    return convert_text_to_robot_ops(PROTOCOL)


def test_brackets_and_citations_are_not_procedure_steps():
    steps = extract_steps(PROTOCOL)
    assert len(steps) == 11
    assert all(s not in {"[", "]"} and "[A1." not in s for s in steps)


def test_numbered_annotation_heading_does_not_hide_later_operations():
    text = (
        "Procedure:\n"
        "1. Define the target using individual particles:\n"
        "   Aspect ratio 6.\n"
        "2. Add 10 mL water to the flask.\n"
        "3. Stir at 500 rpm for 5 minutes."
    )
    steps = extract_steps(text)
    assert steps[1:] == [
        "Add 10 mL water to the flask.",
        "Stir at 500 rpm for 5 minutes.",
    ]
    doc = convert_text_to_robot_ops(text)
    assert [s["action"] for s in doc["steps"]] == ["process", "add_solvent", "stir"]
    assert [op["verb"] for op in doc["micro_plan"]] == [
        "pour",
        "move_to_stir_plate",
        "set_stir_rate",
        "wait",
    ]
    assert doc["_executor"]["valid"] is True


def test_numbered_heading_keeps_its_indented_substeps():
    text = (
        "Procedure:\n"
        "1. Prepare the mixture:\n"
        "   1. Add 10 mL water to the flask.\n"
        "   2. Stir at 500 rpm for 5 minutes.\n"
        "2. Transfer the mixture to a clean beaker."
    )
    steps = extract_steps(text)
    assert steps[0].startswith("Prepare the mixture: Add")
    assert steps[1].startswith("Prepare the mixture: Stir")
    assert steps[2] == "Transfer the mixture to a clean beaker."


def test_unnumbered_section_heading_can_cover_numbered_steps():
    text = "Procedure:\nPreparation:\n1. Add 10 mL water.\n2. Stir for 5 minutes."
    assert extract_steps(text) == [
        "Preparation: Add 10 mL water.",
        "Preparation: Stir for 5 minutes.",
    ]


@pytest.mark.parametrize("strict", [False, True])
def test_exports_never_include_pddl(monkeypatch, strict):
    monkeypatch.setenv("GT_SCHEMA_STRICT", "1" if strict else "0")
    doc = convert_text_to_robot_ops(PROTOCOL)
    assert "generated_pddl" not in doc
    assert doc["_executor"]["schema_version"] == "executor.v1"
    assert doc["_executor"]["valid"] is False
    assert any(op.get("review_required") for op in doc["micro_plan"])
    assert "generated_pddl" not in convert_text_to_gt_schema(PROTOCOL)


def test_all_reaction_additions_precede_stirring_and_workup(screen):
    plan = screen["micro_plan"]
    assert [(op["reagent"], op["volume"]) for op in plan[:3]] == [
        ("DI water", 120),
        ("CTAB/chloroform stock", 15),
        ("metal precursor stocks", 15),
    ]
    spin = next(i for i, op in enumerate(plan) if op["verb"] == "centrifuge")
    reduction_transfer = next(i for i, op in enumerate(plan) if op.get("from") == "V2")
    assert reduction_transfer < spin
    sources = [op["source_step_index"] for op in plan]
    assert sources == sorted(sources)


def test_stir_bar_does_not_create_a_stir_duration():
    doc = convert_text_to_robot_ops(
        "Add 120 mL DI water to the flask containing the stir bar."
    )
    assert [op["verb"] for op in doc["micro_plan"]] == ["pour"]
    assert not doc["timing_delays"]


def test_stirring_has_controls_and_only_explicit_waits(screen):
    plan = screen["micro_plan"]
    assert [op["minutes"] for op in plan if op["verb"] == "wait"] == [30, 30]
    stir = next(i for i, op in enumerate(plan) if op["verb"] == "set_stir_rate")
    assert plan[stir - 1]["verb"] == "move_to_stir_plate"
    assert plan[stir + 1]["verb"] == "wait"
    assert not any(op.get("minutes") == 60 for op in plan)


def test_cold_reductant_uses_a_separate_vessel_and_keeps_deadline(screen):
    plan = screen["micro_plan"]
    borohydride = next(op for op in plan if op.get("reagent") == "NaBH4")
    assert (
        borohydride["vessel"],
        borohydride["amount"],
        borohydride["amount_unit"],
    ) == ("V2", 600, "mg")
    water = next(op for op in plan if op.get("reagent_temperature_C") == 4)
    assert water["vessel"] == "V2" and water["volume"] == 15
    assert not any(
        op["verb"] == "set" and op.get("param") == "temperature_C" for op in plan
    )
    transfer = next(op for op in plan if op.get("from") == "V2")
    assert transfer["to"] == "V1" and transfer["volume"] == 15
    assert transfer["max_delay_minutes"] == 2


def test_four_tubes_three_washes_and_redispersion_are_preserved(screen):
    plan = screen["micro_plan"]
    tubes = {f"V1_tube_{i}" for i in range(1, 5)}
    assert tubes.issubset(screen["vessel_registry"])
    assert {
        op["to"] for op in plan if op["verb"] == "transfer_to_centrifuge_tube"
    } == tubes
    spins = [op for op in plan if op["verb"] == "centrifuge"]
    assert len(spins) == 4
    assert all(
        set(op["tubes"]) == tubes and op["rpm"] == 9000 and op["minutes"] == 5
        for op in spins
    )
    assert {op.get("wash_cycle") for op in spins} == {None, 1, 2, 3}
    for cycle in range(1, 4):
        washes = [
            op for op in plan if op["verb"] == "pour" and op.get("wash_cycle") == cycle
        ]
        assert {op["vessel"] for op in washes} == tubes
        assert all(op["reagent"] == "ethanol" and op["volume"] is None for op in washes)
    assert (
        len(
            [
                op
                for op in plan
                if op.get("purpose") == "final_redispersion" and op["verb"] == "pour"
            ]
        )
        == 4
    )
    assert any(op["verb"] == "decant_supernatant" for op in plan)
    assert any(op["verb"] == "resuspend" for op in plan)


def test_incomplete_screen_requires_review(screen):
    meta = screen["_executor"]
    assert meta["valid"] is False and meta["review_required"] is True
    assert any(
        "quantity" in error or "amount" in error for error in meta["validation_errors"]
    )
    assert any("selection" in error for error in meta["validation_errors"])


def test_materials_are_chemical_names_and_ambiguities_are_reported(screen):
    summary = screen["chemistry_summary"]
    materials = {m["formula_or_short_name"]: m for m in summary["materials"]}
    assert materials["NiCl2"]["concentration"] == 20
    assert materials["H2PtCl6"]["concentration"] == 20
    assert materials["NaBH4"]["amount"] == 600
    assert "CTAB" in materials and "ethanol" in {name.lower() for name in materials}
    assert not any(
        "**" in m["name"] or "Reductant," in m["name"] for m in materials.values()
    )
    assert any("ethanol" in a and "quantity" in a for a in summary["known_ambiguities"])
    assert not any("chloroform volume" in a for a in summary["known_ambiguities"])


def test_fully_specified_simple_protocol_is_valid():
    doc = convert_text_to_robot_ops(
        "Procedure:\n1. Add 10 mL water to the flask.\n2. Stir at 500 rpm for 5 minutes."
    )
    assert doc["_executor"]["valid"] is True
    assert doc["_executor"]["validation_errors"] == []


def test_compound_additions_keep_separate_preparation_and_waits():
    source = (
        Path(__file__).parent / "fixtures" / "synthetic_protocol_variants.txt"
    ).read_text()
    doc = convert_text_to_robot_ops(source)
    plan = doc["micro_plan"]
    assert [(op["reagent"], op["volume"]) for op in plan[:4]] == [
        ("water", 10),
        ("stock A", 2),
        ("stock B", 3),
        ("stock C", 1),
    ]
    assert [op["minutes"] for op in plan if op["verb"] == "wait"] == [5, 8]
    solute = next(op for op in plan if op.get("reagent") == "NaCl")
    assert solute["vessel"] == "V2" and solute["amount"] == 100
    water = next(op for op in plan if op.get("reagent_temperature_C") == 5)
    assert water["vessel"] == "V2" and water["volume"] == 4
    transfer = next(op for op in plan if op.get("from") == "V2")
    assert transfer["to"] == "V1" and transfer["volume"] == 4
    assert transfer["max_delay_minutes"] == 3
    assert not any(op["verb"] == "stir" and op.get("minutes") == 2 for op in plan)
    assert not any(
        op["verb"] == "set" and op.get("param") == "temperature_C" for op in plan
    )
    assert any(
        "temperature_range_requires_selection" in error
        for error in doc["_executor"]["validation_errors"]
    )


def test_later_washes_keep_tube_count_and_explicit_volumes():
    source = (
        Path(__file__).parent / "fixtures" / "synthetic_protocol_variants.txt"
    ).read_text()
    doc = convert_text_to_robot_ops(source)
    tubes = {f"V1_tube_{index}" for index in range(1, 4)}
    spins = [op for op in doc["micro_plan"] if op["verb"] == "centrifuge"]
    assert len(spins) == 3
    assert all(set(op["tubes"]) == tubes for op in spins)
    for cycle in range(1, 3):
        washes = [
            op
            for op in doc["micro_plan"]
            if op["verb"] == "pour" and op.get("wash_cycle") == cycle
        ]
        assert {op["vessel"] for op in washes} == tubes
        assert all(op["reagent"] == "water" and op["volume"] == 3 for op in washes)
    assert doc["_executor"]["valid"] is False


def test_unselected_wait_range_requires_review_without_choosing_endpoint():
    doc = convert_text_to_robot_ops("Procedure:\n1. Allow reduction for 20–30 minutes.")
    assert not doc["micro_plan"]
    assert doc["steps"][0]["minutes_range"] == [20, 30]
    assert any(
        "missing_duration" in error for error in doc["_executor"]["validation_errors"]
    )


def test_ordinary_dissolution_does_not_move_an_existing_reaction():
    doc = convert_text_to_robot_ops(
        "Procedure:\n1. Dissolve 1 g NaCl in 10 mL water.\n2. Stir at 500 rpm for 5 minutes."
    )
    assert all(op.get("vessel", "V1") == "V1" for op in doc["micro_plan"])


def test_validator_detects_out_of_order_and_missing_setpoint():
    doc = {
        "devices": {"hotplate_id": "HP1"},
        "vessel_registry": {"V1": "Flask"},
        "steps": [],
        "micro_plan": [
            {"verb": "wait", "minutes": 1, "source_step_index": 2},
            {
                "verb": "set",
                "device": "HP1",
                "param": "temperature_C",
                "value": None,
                "source_step_index": 1,
            },
        ],
    }
    errors = validate_execution_plan(doc)
    assert any("out of source-step order" in e for e in errors)
    assert any("setpoint" in e for e in errors)


def test_postprocessing_is_idempotent_and_removes_old_pddl(screen):
    old = dict(screen, generated_pddl={"problems": ["obsolete"]})
    repaired = apply_postprocessing(old)
    assert "generated_pddl" not in repaired
    assert repaired["micro_plan"] == screen["micro_plan"]
    assert repaired["micro_plan_min"] == screen["micro_plan_min"]
    assert repaired["timing_delays"] == screen["timing_delays"]


def test_fallback_repairs_stay_with_the_source_step():
    doc = {
        "steps": [
            {
                "action": "stir",
                "raw": "Add 10 mL ethanol to the flask.",
                "vessel": "V1",
                "ops": [{"op": "wait", "minutes": 2}],
            },
            {
                "action": "stir",
                "raw": "Stir for 5 minutes.",
                "vessel": "V1",
                "ops": [{"op": "wait", "minutes": 5}],
            },
        ],
        "vessel_registry": {"V1": "Flask"},
    }
    repaired = apply_postprocessing(doc)
    sources = [op["source_step_index"] for op in repaired["micro_plan"]]
    assert sources == sorted(sources)
    assert any(
        op["verb"] == "pour" and op["source_step_index"] == 1
        for op in repaired["micro_plan"]
    )
