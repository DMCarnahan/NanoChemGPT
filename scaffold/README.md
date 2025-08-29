# NanoChemGPT Scaffold

This scaffold adds **reproducibility plumbing** so reviewers can (a) build environment, (b) rebuild KB/index, (c) run the eval suite, and (d) launch a demo — in a few commands.

## Quickstart

```bash
# 1) Configure environment
cp .env.example .env            

# 2) Create venv and install deps
make setup

# 3) (Optional) Harvest and index knowledge
make kb
make index

# 4) Run evaluations (span / span_attr / struct)
make eval

# 5) Launch the demo server (http://127.0.0.1:8000)
make demo
```

## Make targets

- `make setup` — creates `.venv` and installs from `requirements.txt`.
- `make kb` — runs `harvester/harvester.py --config configs/harvest.yaml` if present.
- `make index` — best-effort index building (adjust to your exact scripts).
- `make eval` — calls `ai_eval/grader.py` over JSON configs: `configs/eval_span.json`, `configs/eval_span_attr.json`, `configs/eval_struct.json`.
- `make demo` — starts Gunicorn with safe defaults.
- `make reproduce` — full pipeline: setup → kb → index → eval → reports.
- `make reports` — lists produced reports in `reports/`.

> Adjust the `kb` and `index` targets to your exact CLI if they differ.

## Environment variables

Create a `.env` from the provided example and adjust as needed.

- `ADMIN_TOKEN` — required to call admin endpoints (e.g., `/admin/rebuild_mech_index`).
- `EMBED_BACKEND` — `openai` or `sentence-transformers`.
- `EMBED_MODEL` — model id for the chosen backend.
- `OPENAI_EMB` — embedding model (if using OpenAI).
- `SPACY_MODEL` — path to your trained spaCy NER (e.g., `harvester/miner/ner_model/model-best`).
- *(Add keys for external services if used:)* `GROBID_URL`, `OPENALEX_BASE`, etc.

## Endpoints

- `GET /healthz` — returns a small JSON status (consider redacting secrets).
- `POST /ask` — unified Q&A endpoint.
- `POST /admin/rebuild_mech_index` — **requires** header `Authorization: Bearer $ADMIN_TOKEN`.

## Notes

- Python **3.11** is assumed for local runs and in the `Dockerfile`.
- If you're on Railway/Render, rely on **runtime env vars** (not `railway.json`). This repo reads `os.environ` at runtime.
- For Python 3.12 + recursive JSON extraction, prefer `regex` package over stdlib `re`.

## Citation & Grounding (for the paper/demo)

- Display a numbered **References** block.
- Extract **used** reference numbers from the generated text (e.g., `[3]`, `[7–9]`) and display only those in the final References section.
- Gate answering with a **sufficiency check** on retrieval quantity/quality; if insufficient, re-search, else answer.
