# NanoChemGPT Copilot Instructions

## Architecture Overview
NanoChemGPT is a domain-specific RAG system for nanochemistry literature mining and synthesis protocol generation. The system follows a **dual-application architecture**:

- **Flask app** (`app.py`): Main web interface and API endpoints for Q&A, protocol conversion, and citations
- **FastAPI wrapper** (`main_asgi.py`): Mounts Flask app and exposes retriever service at `/retriever`

## Core Data Flow
```
User Query → decider/intent.py → judge_sufficiency.py → retriever/ (FAISS search) → LLM reasoning → Citations/Protocols
                ↓ (if insufficient)
            harvester/ → EU-PMC/ArXiv → JSONL → FAISS indexes
```

## Key Components & Responsibilities

### `/decider/` - Intent Classification & Sufficiency
- `intent.py`: Regex-based classification (procedure/comparison/mechanism/definition)
- `judge_sufficiency.py`: Decides whether to trigger new literature mining
- `miner_queue.py`: Enqueues background text-mining jobs

### `/harvester/` - Literature Mining Pipeline
- `harvester.py`: Main ETL orchestrator (EU-PMC/ArXiv → JSONL)
- `miner/`: spaCy NER models for material extraction
- `enhanced_relevance.py`: Multi-layered relevance scoring for harvested papers
- `enhanced_citations.py`: Context-aware citation filtering and ranking
- Config: `harvester/config.yaml` with domain-specific search queries
- Output: `harvester/out_auto/bundle.jsonl` and `bundle_with_methods.jsonl`

### `/retriever/` - Vector Search & FAISS
- `retriever.py`: Multi-index FAISS search (document/passage level)
- `index_jsonl.py`: JSONL → FAISS indexing utilities
- Path resolution via env vars: `RETRIEVER_INDEX_DIRS`, `RETRIEVER_INDEX_DIR_DOC`
- Multiple stores: Uploads, KB, Mechanistic KB

### `/ai_eval/` - Evaluation Harness
- `grader.py`: Span/span_attr/struct evaluation metrics (PRF1, IoU matching)
- `assist_runner.py`: Batch evaluation runner
- Config-driven: `ai_eval/configs/eval_*.yaml` define tasks, datasets, matching rules

## Critical Patterns

### Environment Configuration
- Primary config: `.env` file (copy from `env.example`)
- Key variables: `SPACY_MODEL`, `EMBED_BACKEND`, `OPENAI_EMB`, `RETRIEVER_INDEX_DIRS`
- Path resolution uses fallback hierarchy (explicit → heuristic → defaults)

### JSONL-Centric Data Format
All datasets, mining outputs, and evaluation data use JSONL:
```jsonl
{"text": "...", "entities": [{"label": "MATERIAL", "start": 10, "end": 20}]}
```

### Citation Management (`ref_utils.py`)
- `extract_used_ref_indexes()`: Extracts [1], [2] citations from LLM responses
- `renumber_citations()`: Renumbers citations sequentially
- `format_references_block()`: Generates numbered reference blocks
- Enhanced relevance: `enhanced_citations.py` provides context-aware citation filtering
- Scoring: Query alignment, content quality, context match, authority scores

### spaCy Model Integration
- Custom NER model: `harvester/miner/ner_model/model-best/`
- Labels: ACTION, AMOUNT, ATMOS, CONC, EQUIPMENT, MATERIAL, SPEED, TEMP, TIME, UNIT
- Used in both harvester pipeline and evaluation

## Development Workflows

### Running the Application
```bash
# Development
python app.py  # Flask only
uvicorn main_asgi:app --reload  # FastAPI wrapper

# Production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main_asgi:app
```

### Evaluation Pipeline
```bash
# Run span extraction evaluation
python ai_eval/assist_runner.py \
  --task span \
  --gold ai_eval/datasets/gold_span.jsonl \
  --out ai_eval/runs/results.jsonl \
  --spacy-model harvester/miner/ner_model/model-best

# Generate evaluation report
python ai_eval/grader.py \
  --config ai_eval/configs/eval_span.yaml
```

### Literature Harvesting
```bash
# Mine literature for specific queries
python harvester/harvester.py \
  --config harvester/config.yaml \
  --max-results 200
```

### Vector Index Building
```bash
# Build FAISS indexes from JSONL
python retriever/index_jsonl.py \
  --input harvester/out_auto/bundle.jsonl \
  --output retriever/index_doc/
```

## Code Conventions

### Error Handling Pattern
Graceful degradation with safe fallbacks (see `app.py` lines 51-75):
```python
try:
    from app_utils.helpers import classify_intent
except Exception:
    def classify_intent(q): return "procedure"  # Safe fallback
```

### Path Management
Use `Path` objects and env-based resolution:
```python
ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "/mnt/data/uploads")).resolve()
```

### Async/Background Jobs
Long-running tasks (harvesting, mining) use `enqueue_text_mining_job()` for background processing.

## Integration Points

### MongoDB Integration (`mongo_client.py`)
- Used for persistent storage of queries, results, and user sessions
- Database access via `get_db()` function with connection pooling
- Collections: queries, results, mining_jobs

### Protocol Conversion (`converter.py`)
- Converts free-text synthesis procedures to structured JSON robot operations
- Functions: `validate_step()`, `convert_text_to_robot_ops()`
- Operations: pick_up, pour, place, heat, cool, mix, wait with parameters

<!-- DuckDB integration removed -->

### Mechanistic Reasoning (`mech_reasoning/`)
- Domain-specific mechanistic knowledge base
- Admin routes at `/admin/*` for KB management
- Separate FAISS indexes for mechanistic content

## Windows Development Notes
- Use PowerShell commands in `scripts/powershell eval cmd.sh`
- Path separators handled via `Path` objects for cross-platform compatibility
- Environment variables loaded via `python-dotenv`

## Testing & Debugging
- Evaluation configs in `ai_eval/configs/` define reproducible test scenarios
- Error analysis via `reports/errors_*.jsonl` output
- Use `scripts/powershell eval cmd.sh` for Windows PowerShell evaluation commands
- Metrics: Precision/Recall/F1, IoU matching for span extraction, Brier scores for confidence

## Dependency Management
- Core ML: PyTorch CPU, spaCy 3.7+, sentence-transformers, FAISS
- Web: Flask 3.1+, FastAPI 0.95+, gunicorn for production
- Data: pandas, numpy, duckdb for structured queries
- NLP: transformers, tokenizers, custom spaCy NER model
- Install order matters: torch CPU first, then spaCy, then requirements.txt

## **Enhanced Citation Integration Summary**

### **Both Harvester & Citation Systems Are Now Integrated:**

**🔧 Harvester Relevance (`harvester.py`):**
- ✅ Enhanced relevance filtering during literature mining
- ✅ Multi-dimensional scoring (content, domain, recency, quality, entities)
- ✅ Configurable via `harvester/config.yaml`
- ✅ Relevance metadata stored in JSONL output

**📝 Citation Relevance (`app.py`):**
- ✅ Context-aware citation filtering in response generation
- ✅ Query alignment and content quality scoring
- ✅ Low-relevance citation removal and renumbering
- ✅ Configurable via environment variables

### **Configuration:**

**Harvester Settings (`harvester/config.yaml`):**
```yaml
enable_enhanced_relevance: true
min_year: 2020
quality_threshold: 0.4
max_papers: 150
```

**Citation Settings (`.env`):**
```bash
ENABLE_ENHANCED_CITATIONS=true
CITATION_MIN_SCORE=0.25
```

### **Data Flow:**
```
Search → Enhanced Relevance Filter → Process → Generate Response → Enhanced Citation Filter → Final Output
```

The system now provides end-to-end relevance optimization from literature harvesting through citation generation!

## Common Troubleshooting
- **Missing embeddings**: Check `EMBED_BACKEND` and `OPENAI_EMB` env vars
- **Index not found**: Verify `RETRIEVER_INDEX_DIRS` paths exist and contain FAISS files
- **spaCy model errors**: Ensure `SPACY_MODEL` points to `harvester/miner/ner_model/model-best`
- **JSONL parsing**: All datasets must be valid JSONL with consistent schema
- **Import errors**: Use graceful fallbacks pattern from `app.py` lines 51-75