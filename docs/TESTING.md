# Running tests for NanoChemGPT

This document explains how to run unit and integration tests locally and in CI.

Prerequisites
- Python 3.11
- Recommended: create a virtualenv

Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run unit tests (fast)

```powershell
python -m pytest -q
```

Run a specific test

```powershell
python -m pytest tests/test_nanochemgpt.py::TestUtilityFunctions::test_text_sanitization -q
```

Integration tests

- Integration tests are run in-process using Flask's test client (no external server required).
- If you want to exercise the full end-to-end scripts under `scripts/`, start the app manually first:

```powershell
# Start app in dev mode
python app.py
# In another shell, run the script
python scripts/test_transcribe_integration.py
```

Auto-harvest behavior

- The application can optionally run an automated literature harvest and index rebuild when evidence is thin. This is gated by the environment variable `ENABLE_AUTO_HARVEST` which defaults to off.
- To enable auto-harvest (not recommended in CI), set:

```powershell
$env:ENABLE_AUTO_HARVEST = "1"
```

- Auto-harvest relies on optional dependencies and external services (e.g., `harvester`, `grobid`, `retriever`) and should be enabled only on development hosts where those services are available.

Notes
- Some optional heavy ML and PDF backends (sentence-transformers, faiss, pypdf, pdfminer.six) are optional and used only at runtime. Add them to your environment if you use those features.
- CI (GitHub Actions) will install `requirements-dev.txt` and run tests with coverage.
