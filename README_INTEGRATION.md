Running miner integration tests
===============================

This project includes a miner integration test that exercises the spaCy-based extractor
using a trained model located at `harvester/miner/ner_model/model-best`.

Local setup
-----------

1. Create a Python virtual environment (recommended):

   powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install project dependencies (inside the venv):

   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

3. Install spaCy version compatible with the trained model:

   # The model in this repo was trained with spaCy 3.7.4. Install that version to avoid warnings.
   pip install "spacy==3.7.4"

4. (Optional) If you don't have the `model-best` directory present, place a spaCy pipeline
   directory at `harvester/miner/ner_model/model-best` (this repo includes a model-best entry).

Run the miner integration test
------------------------------

Run only the integration test (it will be skipped if spaCy/model is missing):

powershell
python -m pytest -q tests/test_miner_integration.py -o addopts=

Notes
-----
- CI is configured to set `SPACY_MODEL` to the repo `harvester/miner/ner_model/model-best`
  when present. If you update the model, consider pinning the spaCy version accordingly.
- For deterministic integration runs, the test compares miner output against a small
  expected-output fixture when the model-best is present.
