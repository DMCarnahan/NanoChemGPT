## Railway Deployment

The application can run on Railway using a single service (FastAPI+Flask via `main_asgi:app`).
### Recommended Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| PORT | Service port Railway binds | 8000 |
| OPENAI_API_KEY | OpenAI API key (optional if using local models) | sk-... |
| ENABLE_AUTO_HARVEST | Auto-trigger mining when retrieval thin | true |
| HARVEST_OUT_DIR | Writable harvest output directory | /workspace/harvest_out |
| RETRIEVER_INDEX_DIR_DOC | TF-IDF index directory | /workspace/retriever/index_doc |
| ATTACH_DIR | Attachments directory | /workspace/data/attachments |
| AUTOBUILD_FAISS | Build FAISS (vector) index at startup if bundle available | 1 |
| AUTOBUILD_TFIDF | (Implicit) provided by startup hook (already active) | 1 |
| SPACY_MODEL | Path to spaCy model for NER | harvester/miner/ner_model/model-best |

Mount a persistent volume (Railway: service settings) mapped to `/workspace` so harvested bundles and indexes survive restarts.

### Build & Start Commands

Railway Build Command:
```
pip install -U pip
pip install -r requirements.txt
python -m spacy validate || true
```

Railway Start Command:
```
python main_asgi.py
```

### First Deployment Checklist
1. Set env vars above (at least `OPENAI_API_KEY`, `ENABLE_AUTO_HARVEST`, `HARVEST_OUT_DIR`).
2. Deploy once; check logs for `[preflight]` messages.
3. If `bundle.jsonl` present but no TF-IDF index, startup hook will build it automatically.
4. For FAISS vector search, set `AUTOBUILD_FAISS=1` and redeploy.
5. Hit `/healthz` to view composite status JSON.

### Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `no_index` warnings in retriever responses | TF-IDF/FAISS not built yet | Provide bundle / enable auto-build / run index script manually |
| Permission denied writing harvest | `HARVEST_OUT_DIR` not writable | Point to `/workspace/harvest_out` and ensure volume mounted |
| 500 from retriever early | No index and old version without graceful fallback | Redeploy current version |

# NanoChemGPT

_A domain‑specific RAG system and text‑mining pipeline for nanochemistry synthesis, reasoning, and structured protocol generation._

> **📖 Publication-Ready Documentation**: For comprehensive documentation suitable for academic use, see [README_PUBLICATION.md](README_PUBLICATION.md)

---

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-red.svg)](https://flask.palletsprojects.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-009688.svg)](https://fastapi.tiangolo.com/)
<!-- CI / Release badges -->
[![CI](https://github.com/DMCarnahan/NanoChemGPT/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DMCarnahan/NanoChemGPT/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/DMCarnahan/NanoChemGPT)](https://github.com/DMCarnahan/NanoChemGPT/releases)
[![PyPI - Version](https://img.shields.io/pypi/v/NanoChemGPT?label=PyPI)](https://pypi.org/project/NanoChemGPT/)

**Quick Links:**
- 📚 [Complete Documentation](README_PUBLICATION.md)
- 🚀 [Installation Guide](docs/INSTALLATION.md)
- 📖 [API Reference](docs/API.md)
- 💻 [Examples](examples/)
- 🧪 [Testing](tests/)
- 🤝 [Contributing](CONTRIBUTING.md)
- 📄 [Citation](CITATION.md)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Directory Layout](#directory-layout)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Building Indexes & Datasets](#building-indexes--datasets)
- [Running the App](#running-the-app)
- [API](#api)
 - [Structured Protocol Conversion & Fallback](#structured-protocol-conversion--fallback)
 - [Retriever Permission Fallback](#retriever-permission-fallback)
- [Front‑End](#front-end)
- [Evaluation (ai_eval)](#evaluation-ai_eval)
- [spaCy Models](#spacy-models)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
**NanoChemGPT** is a full‑stack system that:
1) **ingests and mines** nanochemistry literature into structured JSONL datasets;
2) builds **FAISS‑backed** vector indexes for retrieval;
3) exposes a **Flask/FASTAPI** service that answers questions, generates step‑by‑step synthesis protocols ("robot mode"), and provides citation‑grounded reasoning ("reasoning mode"); and
4) includes an **evaluation harness** to measure extraction/structuring quality (span/span_attr/struct) and model utility.

The project supports multiple knowledge stores (Uploads, KB, Mechanistic KB) and uses intent classification and sufficiency checks to decide when to rely on existing data vs. **enqueue text‑mining** jobs.


## Key Features
- **RAG for nanochemistry**: literature‑grounded answers with references.
- **Protocol JSON conversion**: convert free text to fine‑grained actions (e.g., `pick_up`, `pour`, `place`, etc.).
- **Multiple stores**: Uploads vector search, global KB, Mechanistic KB.
- **Text‑miner/Harvester**: EU‑PMC/OpenAlex based harvesting → JSONL corpora → FAISS indexes.
- **Citations**: numbered reference blocks with only the **used** citations.
- **Evaluation**: span/span_attr/struct metrics; reports for analysis.
- **Extensible embeddings**: OpenAI + local `sentence-transformers`.


## Architecture
```
                 ┌──────────────────────────┐
 User/Browser →  │  Flask app (/ask, UI)   │
                 └──────────┬──────────────┘
                            │ calls
                 ┌──────────▼──────────────┐
                 │    decider/ (NLP)       │  classify_intent → judge_sufficiency
                 └──────────┬──────────────┘
                            │ RAG
       ┌────────────────────▼──────────────────────┐
       │  retriever/: uploads, KB, mechanistic     │
       │  (FAISS + embeddings)                     │
       └───────────────┬───────────────┬───────────┘
                       │               │
                ┌──────▼─────┐   ┌─────▼───────┐
                │  KB store  │   │ Mechanistic │  (domain specific)
                └────────────┘   └─────────────┘
                       ▲
                       │ if insufficient
                 ┌─────┴─────────┐
                 │ harvester/    │  (EU‑PMC/OpenAlex → JSONL)
                 └───────────────┘
```


## Directory Layout
> Paths may vary slightly—adjust as needed for your checkout.

```
app/
  app.py                         # Main Flask app (routes, bootstrap)
  app_extensions/
    mechanism_routes.py          # Mechanistic KB routes (/admin/*)
  static/, templates/            # Front‑end HTML/CSS/JS

ai_eval/
  grader.py                      # Metrics: span, span_attr, struct
  assist_runner.py               # Batch assistants eval runner
  datasets/                      # Gold/silver datasets (JSONL)
  reports/                       # Eval reports output

decider/
  kb.py                          # KB interfaces, FAISS paths
  classify_intent.py             # intent classifier
  judge_sufficiency.py           # should we mine more?

harvester/
  harvester.py                   # ETL orchestrator
  eupmc_api.py                   # EU‑PMC search/fetch helpers
  openalex_api.py                # OpenAlex search helpers (optional)
  miner/                         # NER/regex extractors (material amounts, etc.)
    ner_model/model-best/        # spaCy model directory (example)

mechanistic_reasoning/ or mech_reasoning/
  retriever/retriever.py         # Mechanistic FAISS index builder/loader

retriever/
  index_jsonl.py                 # Generic JSONL → FAISS indexer (TF‑IDF too)

vector_store/
  __init__.py                    # FAISS helpers, index dir helpers
  uploads_vector.py              # Uploads store index/search

uploads/
  ...                            # User PDFs/notes, etc.

configs/
  *.yaml                         # Harvester/mining configs

scripts/
  *.sh *.ps1                     # Utility scripts (optional)
```


## Quick Start
### Prerequisites
- Python **3.10+** (3.12 supported). If using Windows, WSL2 is recommended.
- FAISS CPU (`faiss-cpu` wheel) or system FAISS.
- spaCy ≥ 3.5.
- (Optional) `sentence-transformers` for local embeddings.
- OpenAI key if using OpenAI embeddings / chat models.

### Installation
```bash
# clone
git clone <fork-or-origin-url> nanochemgpt
cd nanochemgpt

# create env
python -m venv .venv && source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1

# install
pip install -U pip wheel
pip install -r requirements.txt  # or: pip install -e .

# optional dev tools
pip install -r requirements-dev.txt  # ruff, black, pytest, etc.
```


## Configuration
Create a `.env` file at repo root (the app also reads process env vars):

```env
# --- model/chat ---
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=               # optional, if using a proxy

# --- embeddings ---
EMBED_BACKEND=openai           # openai | sentence-transformers
EMBED_MODEL=text-embedding-3-small   # or sentence-transformers/all-MiniLM-L6-v2

# --- paths ---
DATA_DIR=./data
INDEX_DIR=./data/index
MECH_INDEX_DIR=./data/mech
UPLOADS_DIR=./uploads
KB_INDEX_PATH=./data/index.faiss

# --- spaCy ---
SPACY_MODEL=./harvester/miner/ner_model/model-best

# --- server ---
HOST=0.0.0.0
PORT=8000
FLASK_ENV=development
ADMIN_TOKEN=change-me          # required for /admin/* endpoints
```

The app will fall back to sensible defaults if some variables are missing, but the above is recommended.

## Publishing & Docker

Automated publishing and Docker image builds are provided via GitHub Actions in `.github/workflows/`.


Quick Docker run (build locally):

```powershell
docker build -t nanochemgpt:local .
docker run -p 8000:8000 --env-file .env nanochemgpt:local
```

## Deployment

### Docker

#### Writable Data Directories & Permissions
The application writes uploads, attachments, indexes, and harvested bundles to data directories that default to locations derived from the project root. In some container runtimes (notably when running as a non-root user) you may see permission errors like:

```
PermissionError: [Errno 13] Permission denied: '/home/appuser/app/data'
```

To avoid this, ensure the runtime user owns the data paths. A minimal hardened Dockerfile fragment (multi-stage compatible) is:

```Dockerfile
ARG APP_USER=appuser
RUN adduser --disabled-password --gecos "" ${APP_USER} \
 && mkdir -p /app/data/attachments /app/data/uploads /app/data/indexes \
 && chown -R ${APP_USER}:${APP_USER} /app/data
USER ${APP_USER}

# (Optional) pre-build retriever index if bundle exists
# RUN python retriever/index_jsonl.py --input harvester/out_auto/bundle.jsonl --output retriever/index_doc || true
```

Any directory can be overridden via environment variables at runtime:

| Purpose | Env Var | Default (relative) |
|---------|---------|--------------------|
| Attachments (uploaded PDFs/text) | `ATTACH_DIR` | `data/attachments` |
| Raw uploads (generic) | `UPLOADS_DIR` | `data/uploads` |
| Lookup uploads | `LOOKUP_UPLOAD_DIR` | `data/lookup_uploads` |
| Built‑in seed files | `BUILTIN_DIR` | `builtin` |
| Vector / TF-IDF indexes | `INDEX_DIR` | `retriever/index_doc` |

If a directory is not writable, the app now falls back to a temp path (under the system temp dir) and logs a message starting with `[path-fallback]`. To disable this safety net (e.g., in production where you prefer a hard failure) set:

```
NANOCHEM_DISABLE_DIR_FALLBACK=1
```

#### Preflight Retriever Index Check
On startup a lightweight preflight logs warnings if TF-IDF artifacts are missing. To build them manually:

```
python retriever/index_jsonl.py --input harvester/out_auto/bundle.jsonl --output retriever/index_doc
```

The generated `retriever/index_doc` directory can then be baked into an image layer for faster cold starts.

Secrets required for automation
- `PYPI_API_TOKEN` — PyPI API token for publishing wheels (required to publish automatically).
- `PUBLISH_REPOSITORY_URL` — optional, set to TestPyPI URL to publish there instead.
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` — optional, for pushing images to Docker Hub.
- `GHCR_PAT` — optional, if your org requires a PAT for GHCR instead of `GITHUB_TOKEN`.

Quick verification of workflows
1. Push an annotated tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z`.
2. Open the Actions page and watch `Release (PyPI)` and `Build and publish Docker image` run.
3. Inspect the `dist-artifacts-<tag>` artifact attached to the release workflow run.




## Building Indexes & Datasets
### 1) Harvest literature → JSONL
Use `harvester/harvester.py` with a YAML config.

`configs/example.yaml`:
```yaml
since_year: 2010
max_results_per_source: 500      # must be an integer, not a string
sources: ["eupmc", "openalex"]
query: "nanowire OR nanorod cobalt nickel iron synthesis"
out_dir: data/corpora
```
Run:
```bash
python harvester/harvester.py --config configs/example.yaml
```
This will produce JSONL files in `data/corpora/`.

### 2) Build FAISS indexes
**Global/KB index** from a JSONL:
```bash
python retriever/index_jsonl.py \
  --input data/corpora/mined.jsonl \
  --index_dir data/index
# Produces: data/index/{tfidf.pkl, index.faiss, meta.json}
```
**Mechanistic index** (domain‑specific):
```bash
python mech_reasoning/retriever/retriever.py --build \
  --input data/corpora/mechanistic.jsonl \
  --out_dir data/mech
# Produces: data/mech/mechanistic.faiss
```
**Uploads index** (user uploads folder):
```bash
python -c "from vector_store.uploads_vector import UploadsVectorSearch as U; \
U.from_folder('uploads', index_dir='data/uploads_index')"
```

> Tip: When encountering `UploadsVectorSearch.from_folder() got an unexpected keyword argument 'backend'`,
> pin the correct function signature or pass backends via env vars instead of kwargs.


## Running the App
### Development
```bash
# option A: flask
export FLASK_APP=app/app.py
flask run --host $HOST --port $PORT

# option B: gunicorn
pip install gunicorn
gunicorn -w 2 -b $HOST:$PORT 'app.app:create_app()'

# option C: uvicorn (if parts served via FastAPI inside app)
uvicorn app.app:asgi --host $HOST --port $PORT --reload
```

### Production
Use `gunicorn` or a platform process manager (Railway/Render/Docker). Ensure `.env` is supplied.


## API
### `POST /ask`
Unified Q&A endpoint.

**Request**
```json
{
  "question": "How to make ultrathin CoNi nanowires?",
  "mode": "robot" ,          // "robot" | "reasoning" (optional)
  "top_k": 5,                 // retrieval depth (optional)
  "uploads": true             // include uploads store (optional)
}
```

**Response** (truncated)
```json
{
  "answer": "...",
  "rationale": "...",
## Structured Protocol Conversion & Fallback
When calling `/ask` with `{"mode": "robot"}` (or the UI Robot Mode toggle), the system attempts to convert a free‑text synthesis procedure into a structured JSON under the key `robot_operations`.

### Primary Conversion Flow
1. Extract a procedure block (heuristics over headings like "Procedure", "Synthesis", or numbered sections).
2. Run the text through `convert_text_to_robot_ops()` which:
  - Splits into steps.
  - Detects actions (e.g., pick_up, pour, place, heat, cool, mix, wait, wash, filter, transfer).
  - Normalizes quantities, temperatures, times, pH, and container references.
3. Attach metadata (`meta`) including timing and counts.

### Normalization & Executor Guarantees (New)
After conversion, a late normalization pass enforces invariants to make outputs executor-ready:

- Vessel aliasing: `vN_bottle` → `VN`, `vN_tube_bottle` → `VN_tube` (when known via registry/steps).
- Heating safety: insert or ensure `pick_up`/`place` occurs before any `set` for ovens/hotplates.
- Units augmentation: add missing units to `set` parameters (e.g., `temperature_C`, `stir_rpm`, `rate_mL_per_min`).
- Autotitrator rate: canonicalize variants like `rate_ml_per_min` to `rate_mL_per_min` with unit `mL/min`.
- Idempotent set collapse: remove adjacent duplicate `set` ops across steps while preserving provenance (`collapsed_from_steps`).
- Zero-minute waits are removed.

Following normalization, an executor validation/repair pass attaches `robot_operations._executor`:

- `schema_version`: version string for the executor op schema.
- `valid`: true if the plan is executor-readable after repairs.
- `repairs`: list of applied fixes (e.g., inserted `place` before `set temperature_C`).

When `/ask` returns `robot_operations`, the API also lifts these fields to top-level keys: `executor_valid`, `executor_repairs`, `executor_schema_version` for convenience.

### Quality Heuristic
Some inputs superficially parse but yield low‑information output (e.g., 1–2 generic steps, no distinct actions, or missing ops arrays). A heuristic `_is_poor_robot_doc()` scores the primary parse and—if deemed low quality—forces a fallback salvage pipeline even without an exception. Triggered cases are marked with `meta.fallback_quality_triggered = true`.

### Salvage (Fallback) Pipeline
Activated if the primary conversion throws or quality is poor:
1. Optional section slice: attempt to isolate a subsection like `2.1 Synthesis ...` using a conservative regex (stops at next heading or section boundary).
2. Verb / reagent filtering: retain lines containing synthesis verbs (add, stir, heat, maintain, cool, centrifuge, filter, wash, dry, transfer, dissolve, prepare, adjust, etc.) or quantitative tokens (amounts, temperatures, pH).
3. Hard‑wrap merge: join lines that were visually wrapped in PDFs but belong to the same sentence (no terminal punctuation).
4. Fragment segmentation: cautiously split on sentence ends while avoiding chemical tokens.
5. Pseudo numbering: generate `1.`, `2.`, ... lines ensuring each ends with a period, prepended by a synthetic heading (`**Procedure**`).
6. Re‑parse via the same converter.
7. Compare scores `(num_steps, num_distinct_actions, total_ops)` against the primary parse—replace only if improved.

### Metadata Fields
When fallback runs, the following keys appear in `robot_operations.meta`:
| Field | Meaning |
|-------|---------|
| `fallback_used` | Salvage pipeline executed (and possibly replaced primary). |
| `fallback_primary_error` | Exception message from primary pass (if any). |
| `fallback_slice_used` | True if a subsection slice (e.g., 2.1) was applied. |
| `fallback_quality_triggered` | Heuristic forced salvage despite no exception. |
| `fallback_improved` | Salvage score strictly better than primary. |
| `fallback_old_score` | 3‑tuple (steps, distinct actions, ops) for primary. |
| `fallback_new_score` | 3‑tuple for salvage output. |

These fields enable downstream auditing, analytics, or dataset curation (e.g., filtering only improved protocols).

### Error Propagation
If both primary and fallback fail, `robot_operations` may be absent and an error string is folded into the main response diagnostics (implementation logs a combined `primary:<err>; fallback:<err>` message). Tests ensure graceful degradation without 500 responses solely due to conversion failures.

### Extending
To refine:
- Add verbs to the salvage regex.
- Tune quality thresholds in `_is_poor_robot_doc()` (currently length/action count based).
- Introduce an environment variable gate (future enhancement) to disable heuristic triggers for deterministic evaluation.

### Minimal Primitive Micro Plan (`micro_plan_min`)
Some labs deploy liquid handlers or simple robot arms that can only execute a very small action vocabulary. The converter now automatically derives a reduced primitive plan alongside the full `micro_plan` whenever protocol conversion succeeds.

**Purpose:** Provide an immediately executable action list limited to: `pick_up`, `place`, `pour`, `set` (plus optional synthesized timing as delays) so that low‑capability robots can still run the procedure without needing to interpret higher‑level verbs (e.g., `start`, `stop`, `vortex`, `centrifuge`).

**Generated Keys:**
- `micro_plan_min`: ordered list of primitive actions
- `timing_delays`: (present only if waits were removed) list of objects `{after_index, minutes, original_step_index}` indicating when to pause
- `meta.min_primitive_plan`: boolean flag
- `meta.min_plan_wait_mode`: `delays` (default) or `inline` (if waits mapped to synthetic set ops)
- `meta.min_plan_counts`: summaries `{primitives, delays}`
- `meta.min_plan_warning`: optional warning such as `no_pour_found_but_transfer_ops_present`

**Transformation Rules:**
1. Keep only verbs in `{pick_up, place, pour, set}`.
2. Convert `start` → `set(power=on)`, `stop` → `set(power=off)` for the same device.
3. Drop `wait` actions and encode them in `timing_delays` (unless explicitly kept via env var).
4. Discard auxiliary verbs (`vortex`, `decant_supernatant` etc.) because they have already been decomposed into primitive pours/picks earlier in normalization.
5. Collapse consecutive duplicate primitive actions.
6. Backfill missing `step_index` fields so each primitive knows its source procedural step.
7. Suppress zero‑minute waits entirely.

**Oven Temperature Guarantee:** For any drying/oven step a corresponding `set` action with `param=temperature_C` is preserved in both the full and minimal plans (unless the final step explicitly calls for ambient drying, in which case oven/vacuum ops are filtered out).

**Environment Variables:**
| Variable | Effect | Default |
|----------|--------|---------|
| `DISABLE_MIN_PRIMITIVE_PLAN` | Skip derivation of `micro_plan_min` | off |
| `MIN_PLAN_ALLOW_WAIT` | Keep waits inline as `set(device="scheduler", param="delay_minutes", value=...)` instead of emitting `timing_delays` | off |
| `MIN_PLAN_MAP_GENERIC` | Remap device IDs (HP1→`hotplate`, SP1→`stir_plate`, CF1→`centrifuge`, etc.) for hardware‑agnostic output | off |
| `KEEP_SOLUTION_TERM` | Retain literal word `solution` in normalized reagent `name` (always preserved in `display_name`) | off |

**Example Snippet:**
```jsonc
"micro_plan_min": [
  {"verb": "pick_up", "object": "FeSO4·7H2O", "step_index": 1},
  {"verb": "pour", "from": "FeSO4·7H2O", "to": "Beaker (V1)", "amount": 0.5, "unit": "g", "step_index": 1},
  {"verb": "place", "object": "FeSO4·7H2O", "to": "bench", "step_index": 1},
  {"verb": "pick_up", "object": "deionized water", "step_index": 1},
  {"verb": "pour", "from": "deionized water", "to": "Beaker (V1)", "volume": 25, "volume_units": "mL", "step_index": 1},
  {"verb": "place", "object": "deionized water", "to": "bench", "step_index": 1},
  {"verb": "set", "device": "hotplate", "param": "temperature_C", "value": 60, "step_index": 4}
],
"timing_delays": [
  {"after_index": 7, "minutes": 30, "original_step_index": 4}
]
```

**Validation Tests:**
Automated tests (`tests/test_min_plan.py`, `tests/test_min_plan_enrichment.py`) assert:
- At least one `pour` exists if transfer/add ops were parsed.
- No zero‑minute delays are emitted.
- Drying steps contribute an oven temperature set in the reduced plan.
- Dissolve steps preserve full solute naming (e.g., hydrates like `FeSO4·7H2O`).

If you modify the extraction logic, update or extend these tests to prevent regressions.

#### Provenance & Summary (New)

Two additions enhance auditability and downstream analytics of the reduced plan:

1. `collapsed_from_steps`: Any consecutive duplicate `set` actions (same device + param + value) are merged. The surviving action includes a list of the original `step_index` values under this key so source procedural context is not lost.
2. `meta.min_plan_summary`: Aggregate metrics summarizing the minimal plan contents:
   - `unique_objects`, `unique_devices`
   - Verb counts: `pours`, `sets`, `pick_ups`, `places`
   - `total` primitive actions
   - `has_oven_set` boolean flag (presence of an oven temperature set)

Example excerpt illustrating both:
```jsonc
"micro_plan_min": [
  {"verb": "pick_up", "object": "V1", "step_index": 2},
  {"verb": "pour", "src": "V1", "dst": "R3", "step_index": 2},
  {"verb": "place", "object": "V1", "location": "bench", "step_index": 2},
  {"verb": "set", "device": "HP1", "param": "temperature_C", "value": 80, "collapsed_from_steps": [3,4], "step_index": 3}
],
"meta": {
  "min_plan_summary": {
    "unique_objects": 1,
    "unique_devices": 1,
    "pours": 1,
    "sets": 1,
    "pick_ups": 1,
    "places": 1,
    "total": 4,
    "has_oven_set": false
  }
}
```

#### Reagent Solution Term Retention

Reagent parsing normalizes names by default (dropping a trailing word `solution`) while storing the full textual form in `display_name` and annotating a boolean `is_solution`. To retain the literal word `solution` in the normalized `name`, set:

```
KEEP_SOLUTION_TERM=1
```

This can be useful for workflows where the presence of the word itself drives downstream heuristics or UI labeling.

## Retriever Permission Fallback
When building TF‑IDF indexes lazily (auto‑build on demand) the application may lack permission to create the target directory (e.g., read‑only mounted layer). The function `_ensure_tfidf_index()` now:
1. Attempts `mkdir` on the configured index path.
2. On `PermissionError`, constructs a fallback path under the system temp dir (default `/tmp/nanochem_indexes/<index_name>` or overridden via `RETRIEVER_FALLBACK_DIR`).
3. Logs a warning: `[retriever] permission denied for <orig>; falling back to <fallback>`.
4. Proceeds to build artifacts (`tfidf.pkl` / `tfidf.npz`) there, ensuring retrieval queries still function.

This prevents failures in minimal, locked‑down container images (e.g., distroless or rootless Docker). For production environments where silent fallback is undesirable, you can enforce a hard failure by (future option) disabling the fallback via an environment switch or by asserting writability in a preflight script.

### Observability & Testing
Unit tests simulate a permission denial and assert the fallback directory contains the built index. Monitor logs for the warning above to detect unexpected redirections in staging/production.

### Related Env Vars
| Variable | Purpose | Default |
|----------|---------|---------|
| `RETRIEVER_FALLBACK_DIR` | Root for fallback TF‑IDF index directories | `/tmp/nanochem_indexes` |

If you need to inspect which directory was actually used at runtime, query the log stream or (future improvement) expose a diagnostic endpoint returning the active index paths.

  "references": [
    {"id": 1, "title": "...", "url": "...", "year": 2017},
    {"id": 2, "title": "..."}
  ],
  "used_reference_indexes": [1,2]
}
```

### Admin endpoints
- `POST /admin/rebuild_mech_index` — requires `Authorization: Bearer <ADMIN_TOKEN>`
  - 401 → missing/invalid token; 405 → wrong HTTP method.


## Front‑End
- Static HTML/JS in `templates/` + `static/`.
- **Toggles**: Robot Mode, Reasoning Mode, Convert to JSON, Export JSON.
- **References**: a numbered list; handler should render only the **used** refs.

**Common wiring**
```js
// In app.js
const askBtn = document.getElementById('askBtn');
askBtn?.addEventListener('click', async () => {
  console.log('Ask CLICKED');
  const res = await fetch('/ask', { method: 'POST', body: /* form/json */ });
  const data = await res.json();
  renderAnswer(data.answer);
  renderRefsFromData(data);  // pass the full response object
});
```
> If `renderRefsFromData(json)` is invoked while a function expects `data`, ensure the variable names align. Also verify button IDs exist in the DOM and no other handler consumes the click event.

**Downloads**
If the Convert/Export buttons return `200 OK` but the browser doesn’t download a file, ensure the response sets:
```
Content-Type: application/json
Content-Disposition: attachment; filename="answer.json"
```


## Evaluation (ai_eval)
The repo ships a harness to measure extraction/structuring quality.

### Tasks
- **span**: Did we find the correct **text spans** (entity boundaries)?
- **span_attr**: Span **+ attributes** (e.g., amount units, material type).
- **struct**: Full **structured output** (nested fields, relationships) vs. gold.

### Running
Sample from PowerShell (adapting paths):
```ps1
# sample 1500 lines deterministically
python - <<'PY'
import json, random
random.seed(1337)
src="ai_eval/datasets/gold_span.jsonl"; dst="ai_eval/datasets/gold_span_sample1500.jsonl"
rows=[json.loads(l) for l in open(src,encoding="utf-8")]
random.shuffle(rows)
open(dst,"w",encoding="utf-8").write("\n".join(json.dumps(x,ensure_ascii=False) for x in rows[:1500]))
print("Wrote", dst)
PY

# run the assistant eval
python ai_eval/assist_runner.py `
  --task span `
  --gold datasets/gold_span_sample1500.jsonl `
  --out runs/gpt_spacy_span_1500.jsonl `
  --spacy-model .\my_spacy_chem_ner `
  --model gpt-4o `
  --temperature 0 `
  --retries 3 `
  --fallback-spacy `
  --resume `
  --max-chars 2500

# compute metrics
python ai_eval/grader.py -c configs/eval_span.json
# → Span eval → P=0.XXX R=0.XXX F1=0.XXX. Report: reports/report_span.json
```
**Interpreting results**
- **Precision (P)**: of predicted items, fraction that are correct.
- **Recall (R)**: of gold items, fraction we recovered.
- **F1**: harmonic mean; 1.0 is perfect.

> `span_attr` is stricter than `span`, and `struct` is the strictest (penalizes schema/linkage mistakes). For end‑user protocol quality and reasoning correctness, we watch **struct** and human task success; for mining quality we monitor **span/span_attr**.


## spaCy Models
- Place a trained model at `harvester/miner/ner_model/model-best/` (or set `SPACY_MODEL`).
- To train or update: follow the project spaCy configuration; export to `model-best`.


## Troubleshooting
**Harvester TypeError (euPMC)**
```
TypeError: '<' not supported between instances of 'int' and 'str'
```
→ Ensure `max_results_per_source` is an **integer** in YAML.

**Regex recursion**
```
re.compile(r"\{(?:[^{}]|(?R))*\}")  # fails in stdlib re
```
→ Use the `regex` package (`pip install regex`) and `import regex as re`.

**Uploads index builder signature**
```
UploadsVectorSearch.from_folder() got an unexpected keyword argument 'backend'
```
→ Pass backends via env vars or update the function signature to accept `backend=None`.

**/admin/rebuild_mech_index**
- `401 unauthorized` → provide `Authorization: Bearer $ADMIN_TOKEN`.
- `405 method not allowed` → use **POST**.

**Front‑end buttons do nothing**
- Check the element IDs and event listeners.
- Ensure no earlier script returns `false`/`preventDefault()` incorrectly.
- Verify network tab: the `/ask` request should appear and return JSON.

**Downloads not triggering**
- Add `Content-Disposition: attachment; filename=...` to the response.

## FAQ
**How does the system decide to mine more text?**
`judge_sufficiency` inspects retrieved context volume/quality; if thin, it enqueues a mining job (via `harvester`) with the user’s query to expand the KB.

**Which stores are queried?**
Uploads → KB → Mechanistic, in a configurable order. `top_k` controls depth per store.

**Is it possible to use local embeddings only?**
Yes: set `EMBED_BACKEND=sentence-transformers` and `EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2`.

**Where are eval reports saved?**
`ai_eval/reports/` (configurable). The CLI prints the final path.


## Roadmap
- Better action ontology for protocol JSON (normalize `pick_up`/`pour`/`place` everywhere).
- Stronger citation deduplication and DOI normalization.
- UI polish: modern layout, accessible defaults, robust download flows.
- Richer mechanistic KB and causal graph queries.
- Dockerfile + CI for reproducible deployments.


## Contributing
PRs and issues welcome! Please:
- Run formatters (`ruff`, `black`) and tests (`pytest`) before submitting.
- Keep configs and CLI help up‑to‑date.
- Include small gold examples when adding new extractors.
```