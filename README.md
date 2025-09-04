# NanoChemGPT

_A domain‑specific RAG system and text‑mining pipeline for nanochemistry synthesis, reasoning, and structured protocol generation._

---

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
git clone <your-fork-or-origin> nanochemgpt
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

> Tip: If you see `UploadsVectorSearch.from_folder() got an unexpected keyword argument 'backend'`,
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
Use `gunicorn` or your platform’s process manager (Railway/Render/Docker). Ensure `.env` is supplied.


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
> If you mistakenly call `renderRefsFromData(json)` but your function expects `data`, make sure the variable names align. Also verify button IDs exist in the DOM and no other handler swallows the click.

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
- Place your trained model at `harvester/miner/ner_model/model-best/` (or set `SPACY_MODEL`).
- To train/update: follow your project’s spaCy config; export to `model-best`.


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

**Can I use local embeddings only?**
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