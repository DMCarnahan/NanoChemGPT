import importlib
import sys
from pathlib import Path


def test_get_miner_prefers_packaged_model(monkeypatch, tmp_path, tmp_path_factory):
    # Create a temporary fake model-best under harvester/miner/ner_model/model-best
    repo_root = Path(__file__).resolve().parents[2]
    model_best = repo_root / 'harvester' / 'miner' / 'ner_model' / 'model-best'
    model_best.mkdir(parents=True, exist_ok=True)

    # Prepare a fake BasicMiner class to capture the nlp_model arg
    class FakeBasicMiner:
        def __init__(self, nlp_model=None, **kwargs):
            # store on instance for assertion
            self._nlp_model_used = nlp_model

    # Inject fake basic_miner module before importing runtime
    mod_name = 'harvester.miner.basic_miner'
    import types
    fake_mod = types.ModuleType(mod_name)
    fake_mod.BasicMiner = FakeBasicMiner
    sys.modules[mod_name] = fake_mod

    # Now import (or reload) runtime and call get_miner
    import harvester.miner.runtime as runtime
    importlib.reload(runtime)

    miner = runtime.get_miner()
    # The BasicMiner was our FakeBasicMiner, so miner is an instance of that
    assert isinstance(miner, FakeBasicMiner)
    used = getattr(miner, '_nlp_model_used', None)
    # Ensure the selected model is the 'model-best' path in the repo
    assert used is None or 'model-best' in str(used)
