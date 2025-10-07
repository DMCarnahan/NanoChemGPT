# Nanochem Retriever (Hybrid: TF‑IDF + Embeddings)

This is a drop‑in upgrade of the earlier retriever. It adds **semantic embeddings** with two backends:
- **OpenAI** (`text-embedding-3-large` by default)
- **Sentence‑Transformers** (local, default `allenai-specter`)

The service can run **TF‑IDF**, **Embeddings**, or a **Hybrid** (alpha*embedding + (1-alpha)*tfidf).

## Setup
```bash
pip install fastapi uvicorn scikit-learn numpy pydantic==1.*
# Optional backends:
pip install openai>=1.30.0         # for OpenAI embeddings (requires OPENAI_API_KEY)
pip install sentence-transformers   # for local embeddings
```

## 1) Build the index from a harvested bundle
```bash
python index_jsonl.py --bundle ../nanochem_harvester/out/bundle.jsonl --index_dir ./index   --embed-backend openai --embed-model text-embedding-3-large   
```

Environment (if using OpenAI):
```bash
$env:OPENAI_API_KEY = "sk-..."    # PowerShell
# or: export OPENAI_API_KEY=sk-... (bash)
```

## 2) Start the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
# OpenAPI: http://localhost:8000/docs
```

## 3) Query from a GPT Action
- Use `POST /search` with JSON:
```json
{"query":"hot-injection synthesis of PbS quantum dots","k":5,"mode":"hybrid","alpha":0.7}
```
- Modes: `tfidf` | `embed` | `hybrid`
- For `embed`, specify the backend/model when building the index.

## Notes
- Index artifacts live in `index/`:
  - `tfidf.pkl` (TF‑IDF vectorizer, matrix, metadata)
  - `embed.pkl` (embedding config and matrix) if built
- If an embedding index isn't present, the API will fall back to TF‑IDF.
