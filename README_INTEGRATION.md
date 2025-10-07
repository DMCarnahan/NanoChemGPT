README - Integration & CI notes
================================

This file documents how CI obtains the spaCy model used by the miner integration tests, how the retriever uses a safe default index directory, and how to run tests locally in a reproducible way.

Model availability in CI
------------------------
- The repository may include a spaCy model under `harvester/miner/ner_model/model-best`. When present in the checked-out branch, CI will prefer this model and set `SPACY_MODEL` to that path for the miner-integration job.
- If the model directory is not present in the runner, the CI workflow falls back to downloading `en_core_web_sm` (spaCy small) and uses it for integration tests.
- For teams that prefer not to commit model artifacts to the repository, consider one of:
  - Hosting the model files in a release or S3 and adding a workflow step to download and extract them in the job.
  - Storing model files via Git LFS and ensuring the runner has LFS enabled.

Retriever index directory behavior (important for CI)
---------------------------------------------------
- The retriever service no longer assumes `/data/vector_store` by default. Instead it:
  - Prefers `INDEX_DIR` environment variable if set.
  - Falls back to a repo-local `vector_store/` directory under the project root.
  - If creation of those directories is blocked (PermissionError), the code falls back to a temporary directory (created via Python's `tempfile.mkdtemp`).

This prevents PermissionError/FileNotFoundError during pytest collection in CI where `/data` is not writable.

Running tests locally (recommended reproducible steps)
----------------------------------------------------
Below are the commands to create an isolated virtualenv, install pinned dependencies similar to CI, and run the tests. Adjust Python path if needed.

Windows PowerShell example:

```powershell
# create venv
python -m venv .venv-test
.\.venv-test\Scripts\Activate.ps1

# upgrade pip and install test deps (pin numpy<2 to avoid local faiss ABI issues)
python -m pip install --upgrade pip
pip install "numpy<2" pytest pytest-cov spacy==3.7.4

# (optional) install faiss-cpu if you want miner-integration to use faiss
# pip install faiss-cpu

# run tests (fast subset)
pytest -q -k "not slow" --maxfail=1

# run full suite
pytest -q
```

Notes
-----
- When using a locally compiled `faiss` built against an older NumPy, install `numpy<2` in the virtual environment to avoid `_ARRAY_API` incompatibilities.
- CI uses pinned versions in the miner-integration job; check `.github/workflows/ci.yml` for the exact setup.

An explicit workflow step can be added (on request) to download the model from a release or object storage and keep large model files out of the repository.
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

4. (Optional) If the `model-best` directory is absent, place a spaCy pipeline
   directory at `harvester/miner/ner_model/model-best` (this repo includes a model-best entry).

Run the miner integration test
------------------------------

Run only the integration test (it will be skipped if spaCy/model is missing):

powershell
python -m pytest -q tests/test_miner_integration.py -o addopts=

Notes
-----
- CI is configured to set `SPACY_MODEL` to the repo `harvester/miner/ner_model/model-best`
   when present. When the model is updated, consider pinning the spaCy version accordingly.
- For deterministic integration runs, the test compares miner output against a small
  expected-output fixture when the model-best is present.
