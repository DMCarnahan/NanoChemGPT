import re
from converter import parse_reagent_phrase_to_struct, convert_text_to_robot_ops


def test_solution_flag_and_name_preserved():
    phrase = "0.45 M sodium hydroxide (NaOH) solution"
    struct = parse_reagent_phrase_to_struct(phrase)
    assert struct.get("is_solution") is True, "Expected is_solution flag True"
    # display_name should contain full phrase minus trailing metadata
    assert "sodium hydroxide" in struct.get(
        "display_name", ""
    ), "Full reagent name lost"
    assert struct.get("name").startswith(
        "sodium hydroxide"
    ), "Stripped name missing base chemical"


def test_duplicate_set_collapse_in_min_plan():
    text = (
        "1. **Procedure**:\n"
        "1. Heat the mixture to 60 C for 10 minutes.\n"
        "2. Maintain the temperature at 60 C for 20 minutes.\n"
    )
    doc = convert_text_to_robot_ops(text)
    sets = [
        a
        for a in doc.get("micro_plan_min", [])
        if a.get("verb") == "set"
        and a.get("param") == "temperature_C"
        and a.get("value") == 60
    ]
    # Should only have one temperature_C set in minimal plan after collapse
    assert (
        len(sets) == 1
    ), f"Expected 1 collapsed temperature set, found {len(sets)}: {sets}"
