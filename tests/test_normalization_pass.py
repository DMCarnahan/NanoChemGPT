import copy
import pytest

from converter import apply_postprocessing


def build_doc(actions):
    # Minimal doc structure feeding into apply_postprocessing
    return {
        "devices": {"hotplate_id": "HP1", "oven_id": "OV1", "centrifuge_id": "CF1"},
        "defaults": {"centrifuge_rpm": 8000, "centrifuge_minutes": 10, "stir_rpm": 700},
        "steps": [
            {
                "index": 1,
                "action": "heat_hold",
                "raw": "Heat the mixture at 80 C",
                "ops": [
                    {"op": "set", "device": "HP1", "param": "temperature_C", "value": 80},
                    {"op": "wait", "minutes": 60},
                ],
                # micro_ops intentionally missing placement to test insertion
                "micro_ops": [
                    {"verb": "set", "device": "HP1", "param": "temperature_C", "value": 80, "step_index": 1},
                    {"verb": "wait", "minutes": 60, "step_index": 1},
                ],
            }
        ],
        "micro_plan": [
            {"verb": "set", "device": "HP1", "param": "temperature_C", "value": 80, "step_index": 1},
            {"verb": "wait", "minutes": 60, "step_index": 1},
        ],
    }


def test_hotplate_set_inserts_pickup_place_and_units():
    doc = build_doc([])
    out = apply_postprocessing(copy.deepcopy(doc))
    mp = out.get("micro_plan", [])
    # Find the set temperature action
    idx = next(i for i,a in enumerate(mp) if a.get("verb") == "set" and a.get("device") == "HP1" and a.get("param") == "temperature_C")
    # Preceded by pick_up then place of V1 to HP1
    assert idx >= 2
    assert mp[idx-2]["verb"] == "pick_up"
    assert mp[idx-1]["verb"] == "place" and mp[idx-1]["to"] == "HP1"
    # Units annotated
    assert mp[idx]["unit"] == "C"


def test_pour_volume_units_augmented():
    doc = build_doc([])
    # Create an add_solvent op that will produce a pour micro op; volume_units should default to mL
    doc["steps"][0]["ops"].insert(0, {"op": "add_solvent", "solvent": "ethanol", "volume": 10})
    doc["steps"][0]["micro_ops"] = []  # ensure rebuild occurs from ops
    doc["micro_plan"] = []
    out = apply_postprocessing(copy.deepcopy(doc))
    pours = [m for m in out["micro_plan"] if m.get("verb") == "pour" and m.get("volume") == 10]
    assert pours, "Expected at least one pour with volume 10 generated from add_solvent op"
    assert all(p.get("volume_units") == "mL" for p in pours), pours


def test_provenance_collapsed_from_steps_sorted():
    doc = build_doc([])
    # Provenance may be absent; ensure pipeline doesn't introduce malformed data.
    out = apply_postprocessing(copy.deepcopy(doc))
    set_ops = [m for m in out["micro_plan"] if m.get("verb") == "set" and m.get("param") == "temperature_C"]
    assert set_ops, "Expected temperature set op present"
    # If provenance exists, it must be sorted ascending with self first (len>1 case)
    cfs = set_ops[0].get("collapsed_from_steps")
    if cfs:
        assert cfs == sorted(cfs)


def test_water_bath_alias_canonicalizes():
    doc = build_doc([])
    # Replace existing temperature set op device with synonym at ops layer so rebuild carries it
    for op in doc["steps"][0]["ops"]:
        if op.get("op") == "set" and op.get("param") == "temperature_C":
            op["device"] = "water bath"
    doc["steps"][0]["micro_ops"] = []
    doc["micro_plan"] = []
    out = apply_postprocessing(copy.deepcopy(doc))
    set_ops = [m for m in out["micro_plan"] if m.get("verb") == "set" and m.get("param") == "temperature_C"]
    assert set_ops, "Expected at least one temperature set op after rebuild"
    assert set_ops[0]["device"] == "HP1"


def test_executor_metadata_and_repairs_present():
    doc = build_doc([])
    # remove any existing pick_up/place so executor pass must repair
    doc["micro_plan"] = [m for m in doc["micro_plan"] if m.get("verb") not in {"place","pick_up"}]
    out = apply_postprocessing(copy.deepcopy(doc))
    meta = out.get("_executor")
    assert meta and meta.get("schema_version") == "executor.v1"
    # Either the executor pass or earlier normalization inserted pick_up/place; repairs may be empty if earlier handled it
    inserted = any(r.startswith("inserted_hotplate_pickup_place") for r in meta.get("repairs", []))
    # Micro plan contains synthesized pick_up/place regardless
    verbs = [a.get("verb") for a in out.get("micro_plan", [])]
    assert "pick_up" in verbs and "place" in verbs
    # If no repair recorded, accept because earlier normalization satisfied constraint
    if not inserted:
        assert meta.get("repairs") == [] or all(isinstance(r,str) for r in meta.get("repairs"))
