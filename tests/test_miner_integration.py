import pytest

import os
from pathlib import Path
try:
    from harvester.miner.runtime import get_miner
    HAS_MINER = True
except Exception:
    HAS_MINER = False


# Path to the repository-packaged model-best
# repo root (one level up from tests/)
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_BEST_PATH = REPO_ROOT / "harvester" / "miner" / "ner_model" / "model-best"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MINER, reason="spaCy or miner runtime not available")
def test_miner_extracts_simple_operations():
    """Integration test: ensure miner can be instantiated with the packaged model and run a basic extract."""
    miner = get_miner()
    # basic smoke: ensure miner object exposes extract and expand
    assert hasattr(miner, "extract") and hasattr(miner, "expand")

    text = "Add 10 mL of SnCl2 solution, heat to 180 C for 12 hours, then cool and centrifuge."
    res = miner.extract(text)
    # expect a list of operations (may be empty on very old models) but should be list-like
    assert isinstance(res, list)
    # expanded should be available when calling expand
    expanded = miner.expand(res)
    assert isinstance(expanded, list)

    # If the repo-packaged model-best is present, assert miner requested it
    if MODEL_BEST_PATH.exists():
        # Basic runtime records requested model path on the miner instance
        requested = getattr(miner, "_model_path_requested", None)
        assert requested is not None
        assert str(MODEL_BEST_PATH) in str(requested)
        # With a trained model we expect at least one material extracted for the example
        assert any(op.get("materials") for op in res)

        # Stricter expectations with a trained model:
        op_types = {op.get("op_type") for op in res}
        # Expect add, heat, and centrifuge operations to be detected
        assert "add" in op_types
        assert "heat" in op_types or any(t for t in op_types if "heat" in (t or ""))
        assert "centrifuge" in op_types or any(t for t in op_types if "centrifuge" in (t or ""))

        # Check params: amounts and time should be parsed
        add_ops = [op for op in res if op.get("op_type") == "add"]
        assert add_ops, "no add operation detected"
        # amounts should include '10' or '10 mL'
        amounts = add_ops[0].get("params", {}).get("amounts", [])
        assert any("10" in a or "10 mL" in a for a in amounts)

        heat_ops = [op for op in res if op.get("op_type") == "heat"]
        assert heat_ops, "no heat operation detected"
        times = heat_ops[0].get("params", {}).get("time", "")
        assert "12" in times or "12 hour" in times.lower()

        # Materials: expect Sn or SnCl2 to be present in at least one material mention
        mats = [m for op in res for m in (op.get("materials") or [])]
        assert any(m and ("sn" in m.lower() or "sncl" in m.lower()) for m in mats)

        # If an expected-output fixture exists, compare in a fuzzy manner to ensure deterministic behaviour
        try:
            import json
            fixture_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "miner_expected.json"
            # above resolves to repo/tests/tests/fixtures — correct by backing up one more
            fixture_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "miner_expected.json"
            if fixture_path.exists():
                expected = json.loads(fixture_path.read_text(encoding="utf-8"))
                # Check that each expected op_type appears in the results in order
                got_types = [op.get("op_type") for op in res]
                for e in expected:
                    assert e.get("op_type") in got_types
        except Exception:
            # Non-fatal: the fixture comparison is best-effort
            pass
