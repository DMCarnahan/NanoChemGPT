# API Documentation

## Overview

NanoChemGPT provides a RESTful API for nanochemistry literature mining, synthesis protocol generation, and mechanistic reasoning. The API is built on Flask with FastAPI wrapper support for production deployments.

## Base URL

- **Development**: `http://localhost:5000`
- **Production**: `http://localhost:8000`

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing API key authentication.

## Content Types

- **Request**: `application/json` or `multipart/form-data` (for file uploads)
- **Response**: `application/json`

## Error Handling

All endpoints return errors in the following format:

```json
{
  "ok": false,
  "error": "Error description",
  "code": "ERROR_CODE"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found
- `500`: Internal Server Error

## Endpoints

### Health Check

#### GET `/health`

Check system health and availability.

**Response**:
```json
{
  "status": "ok",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

### Question Answering

#### POST `/ask`

Main endpoint for question answering with retrieval-augmented generation.

**Request Body** (JSON):
```json
{
  "question": "How to synthesize gold nanoparticles?",
  "mode": "protocol",
  "intent": "synthesis",
  "k_doc": 5,
  "k_passage": 10,
  "retrieval": "both",
  "allow_fetch": true,
  "want_inline": true,
  "kb_k": 5,
  "web_k": 10,
  "w_doc": 0.6,
  "w_passage": 0.4
}
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | Yes | - | The question to answer |
| `mode` | string | No | `"protocol"` | Response mode: `"protocol"` or `"reasoning"` |
| `intent` | string | No | `"protocol"` | Query intent: `"protocol"`, `"synthesis"`, `"reasoning"`, `"analysis"` |
| `k_doc` | integer | No | 5 | Number of documents to retrieve |
| `k_passage` | integer | No | 10 | Number of passages to retrieve |
| `retrieval` | string | No | `"both"` | Retrieval level: `"doc"`, `"passage"`, or `"both"` |
| `allow_fetch` | boolean | No | `true` | Allow background literature fetching |
| `want_inline` | boolean | No | `true` | Include inline citations |
| `kb_k` | integer | No | 5 | Knowledge base retrieval count |
| `web_k` | integer | No | 10 | Web search retrieval count |
| `w_doc` | float | No | 0.6 | Document weight in combined retrieval |
| `w_passage` | float | No | 0.4 | Passage weight in combined retrieval |

**Response**:
```json
{
  "ok": true,
  "answer": "## Gold Nanoparticle Synthesis Protocol\n\n1. Dissolve 0.1 mmol HAuCl₄·3H₂O in 10 mL deionized water [1].\n2. Add 0.5 mmol sodium citrate as reducing agent [2].\n3. Heat to 100°C while stirring for 15 minutes [1].\n4. Cool to room temperature and centrifuge at 8000 rpm [3].",
  "rationale": "This protocol uses the Turkevich method for gold nanoparticle synthesis. Sodium citrate acts as both reducing agent and stabilizer, preventing aggregation while controlling particle size through nucleation kinetics.",
  "refs": [
    {
      "title": "Synthesis and Size Control of Gold Nanoparticles",
      "authors": "Turkevich, J. et al.",
      "journal": "Faraday Discussions",
      "year": 1951,
      "doi": "10.1039/df9511100055",
      "url": "https://doi.org/10.1039/df9511100055"
    },
    {
      "title": "Citrate-Stabilized Gold Nanoparticles",
      "authors": "Frens, G.",
      "journal": "Nature Physical Science",
      "year": 1973,
      "doi": "10.1038/physci241020a0",
      "url": "https://doi.org/10.1038/physci241020a0"
    }
  ],
  "usage_markers": ["protocol", "temperature_control", "citrate_reduction"],
  "context_stats": {
    "attachments_chars": 0,
    "uploads_chars": 2456,
    "kb_chars": 1875,
    "web_chars": 3421
  }
}
```

#### POST `/ask` (with file upload)

Submit questions with file attachments for context.

**Request** (multipart/form-data):
- `question`: The question text
- `file`: PDF or text file attachment
- Additional parameters as form fields

**Example using curl**:
```bash
curl -X POST "http://localhost:5000/ask" \
  -F "question=Analyze this synthesis protocol and suggest optimizations" \
  -F "file=@synthesis_protocol.pdf" \
  -F "mode=reasoning"
```

---

### Protocol Transcription

#### POST `/transcribe`

Transcribe methods paragraphs from attached papers into robot-mode format without changing the content itself. This endpoint intelligently extracts the most relevant method section and structures it for the converter module.

**Request** (multipart/form-data or JSON):

**For file upload**:
```bash
curl -X POST "http://localhost:5000/transcribe" \
  -F "file=@synthesis_paper.pdf" \
  -F "convert_to_robot=true"
```

**For direct text**:
```json
{
  "text": "Heat 100 mL of 0.5 mM HAuCl₄ solution to boiling. Add 10 mL of 38.8 mM sodium citrate solution rapidly. Continue boiling for 15 minutes while stirring.",
  "convert_to_robot": true
}
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | No* | - | PDF or text file containing method description |
| `text` | string | No* | - | Direct text input for transcription |
| `convert_to_robot` | boolean | No | `false` | Whether to also convert to robot operations |

*Either `file` or `text` must be provided.

**Response**:
```json
{
  "ok": true,
  "original_text": "Heat 100 mL of 0.5 mM HAuCl₄ solution to boiling...",
  "structured_protocol": "## Synthesis Protocol:\n\n1. **Hardware & Glassware**:\n   - Water bath (100°C)\n\n2. **Materials**:\n   - iron(II) sulfate heptahydrate (FeSO4·7H2O) 0.5 mM, 100 mL (iron oxide precursor)\n   - sodium hydroxide (NaOH) 38.8 mM (precipitating agent)\n\n3. **Procedure**:\n   1. Prepare the iron(II) sulfate solution (0.5 mM), 100 mL.\n   2. Maintain the reaction in a water bath at 100°C with continuous stirring.\n   3. Add NaOH (38.8 mM) to the iron solution, monitoring pH.",
  "extracted_facts": {
    "hardware": ["Water bath (100°C)"],
    "materials": [
      {
        "name": "iron(II) sulfate heptahydrate (FeSO4·7H2O)",
        "concentration": "0.5 mM",
        "volume": "100 mL",
        "role": "iron oxide precursor"
      }
    ],
    "procedure": [
      "Prepare the iron(II) sulfate solution (0.5 mM), 100 mL.",
      "Maintain the reaction in a water bath at 100°C with continuous stirring."
    ]
  },
  "robot_operations": {
    "steps": [
      {
        "action": "heat",
        "target": "solution",
        "temperature": 100,
        "temperature_unit": "°C"
      },
      {
        "action": "add",
        "source": "sodium citrate",
        "target": "solution",
        "volume": 10,
        "volume_unit": "mL",
        "method": "rapid"
      }
    ]
  }
}
```

**Key Features**:
- **Intelligent Extraction**: Uses `_pick_method_paragraph()` to automatically identify the most relevant method section from uploaded papers
- **Content Preservation**: Maintains original content while structuring it for robot operations
- **Dual Output**: Provides both human-readable structured protocol and machine-readable robot operations
- **File Support**: Handles PDF and text file uploads with automatic text extraction
- **Fact Extraction**: Identifies specific materials, equipment, and procedural steps

**Use Cases**:
- Converting literature protocols to robot-executable formats
- Structuring experimental procedures for laboratory automation
- Extracting synthesis conditions from research papers
- Preparing protocols for the converter module

---

### Protocol Conversion

#### POST `/convert`

Convert free-text synthesis protocols to structured robot operations.

**Request Body**:
```json
{
  "text": "Heat the solution to 80°C and stir for 2 hours. Then add 5 mL of reducing agent dropwise over 10 minutes.",
  "target_ops": ["heat", "mix", "add", "wait"],
  "validate": true
}
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Free-text protocol to convert |
| `target_ops` | array | No | `[]` | Expected operation types for validation |
| `validate` | boolean | No | `true` | Validate generated operations |

**Response**:
```json
{
  "ok": true,
  "operations": [
    {
      "action": "heat",
      "target": "solution",
      "temperature": 80,
      "temperature_unit": "°C",
      "step": 1
    },
    {
      "action": "mix",
      "target": "solution",
      "method": "stir",
      "step": 2
    },
    {
      "action": "wait",
      "duration": 2,
      "duration_unit": "hours",
      "step": 2
    },
    {
      "action": "add",
      "source": "reducing agent",
      "target": "solution",
      "volume": 5,
      "volume_unit": "mL",
      "method": "dropwise",
      "rate": "10 minutes",
      "step": 3
    }
  ],
  "validation": {
    "valid": true,
    "warnings": [],
    "missing_ops": []
  },
  "text_analysis": {
    "entities": [
      {"type": "TEMP", "value": "80°C", "span": [19, 23]},
      {"type": "TIME", "value": "2 hours", "span": [42, 49]},
      {"type": "VOLUME", "value": "5 mL", "span": [60, 64]}
    ]
  }
}
```

---

### Upload Management

#### POST `/upload`

Upload documents for processing and indexing.

**Request** (multipart/form-data):
- `file`: Document file (PDF, TXT, DOCX)
- `title`: Optional document title
- `tags`: Optional comma-separated tags

**Response**:
```json
{
  "ok": true,
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "synthesis_protocol.pdf",
  "size": 245760,
  "pages": 12,
  "processed": true,
  "index_status": "indexed"
}
```

#### GET `/uploads`

List all uploaded documents.

**Response**:
```json
{
  "ok": true,
  "uploads": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "synthesis_protocol.pdf",
      "title": "Gold Nanoparticle Synthesis",
      "size": 245760,
      "upload_date": "2024-01-01T12:00:00Z",
      "tags": ["synthesis", "gold", "nanoparticles"],
      "indexed": true
    }
  ],
  "total": 1
}
```

#### DELETE `/upload/<upload_id>`

Delete an uploaded document.

**Response**:
```json
{
  "ok": true,
  "message": "Upload deleted successfully"
}
```

---

### Literature Mining

#### POST `/mine`

Trigger background literature mining for specific queries.

**Request Body**:
```json
{
  "query": "gold nanoparticle synthesis citrate reduction",
  "max_results": 50,
  "sources": ["eupmc", "arxiv"],
  "min_year": 2020,
  "filters": {
    "open_access": true,
    "peer_reviewed": true
  }
}
```

**Response**:
```json
{
  "ok": true,
  "job_id": "mining_job_12345",
  "status": "queued",
  "estimated_time": "5-10 minutes",
  "query": "gold nanoparticle synthesis citrate reduction"
}
```

#### GET `/mine/<job_id>`

Check mining job status.

**Response**:
```json
{
  "ok": true,
  "job_id": "mining_job_12345",
  "status": "completed",
  "progress": 100,
  "results": {
    "papers_found": 47,
    "papers_processed": 47,
    "entities_extracted": 342,
    "index_updated": true
  },
  "started_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:08:30Z"
}
```

---

### Knowledge Base Management

#### GET `/kb/search`

Search the knowledge base directly.

**Query Parameters**:
- `q`: Search query
- `k`: Number of results (default: 10)
- `threshold`: Similarity threshold (default: 0.7)

**Response**:
```json
{
  "ok": true,
  "results": [
    {
      "text": "Gold nanoparticles can be synthesized using the Turkevich method...",
      "score": 0.89,
      "source": "doi:10.1039/df9511100055",
      "title": "Synthesis and Size Control of Gold Nanoparticles"
    }
  ],
  "query": "gold nanoparticle synthesis",
  "total_results": 15
}
```

#### POST `/kb/add`

Add content to the knowledge base.

**Request Body**:
```json
{
  "text": "Novel synthesis method for gold nanoparticles using green chemistry principles...",
  "title": "Green Synthesis of Gold Nanoparticles",
  "metadata": {
    "doi": "10.1021/example.2024.123456",
    "authors": ["Smith, J.", "Doe, A."],
    "year": 2024,
    "journal": "Green Chemistry"
  },
  "entities": [
    {"label": "MATERIAL", "start": 35, "end": 52, "text": "gold nanoparticles"},
    {"label": "METHOD", "start": 59, "end": 85, "text": "green chemistry principles"}
  ]
}
```

**Response**:
```json
{
  "ok": true,
  "id": "kb_entry_67890",
  "indexed": true,
  "message": "Content added to knowledge base"
}
```

---

### Evaluation and Metrics

#### POST `/evaluate`

Run evaluation on provided datasets.

**Request Body**:
```json
{
  "task": "span",
  "dataset": "gold_span.jsonl",
  "model": "harvester/miner/ner_model/model-best",
  "metrics": ["precision", "recall", "f1"],
  "config": {
    "match_threshold": 0.5,
    "entity_types": ["MATERIAL", "TEMP", "TIME", "EQUIPMENT"]
  }
}
```

**Response**:
```json
{
  "ok": true,
  "evaluation_id": "eval_12345",
  "status": "running",
  "task": "span",
  "estimated_time": "2-3 minutes"
}
```

#### GET `/evaluate/<evaluation_id>`

Get evaluation results.

**Response**:
```json
{
  "ok": true,
  "evaluation_id": "eval_12345",
  "status": "completed",
  "results": {
    "overall": {
      "precision": 0.87,
      "recall": 0.82,
      "f1": 0.84
    },
    "by_entity": {
      "MATERIAL": {"precision": 0.91, "recall": 0.89, "f1": 0.90},
      "TEMP": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
      "TIME": {"precision": 0.83, "recall": 0.80, "f1": 0.82}
    },
    "confusion_matrix": {
      "MATERIAL": {"tp": 89, "fp": 9, "fn": 11},
      "TEMP": {"tp": 78, "fp": 14, "fn": 22}
    }
  },
  "dataset_size": 200,
  "processing_time": "2m 15s"
}
```

---

### System Information

#### GET `/info`

Get system information and capabilities.

**Response**:
```json
{
  "ok": true,
  "version": "1.0.0",
  "capabilities": {
    "vector_search": true,
    "file_upload": true,
    "literature_mining": true,
    "protocol_conversion": true,
    "mechanistic_reasoning": true,
    "evaluation": true
  },
  "models": {
    "embedding": "text-embedding-3-small",
    "llm": "gpt-4",
    "ner": "harvester/miner/ner_model/model-best"
  },
  "index_stats": {
    "document_count": 15420,
    "passage_count": 142380,
    "last_updated": "2024-01-01T10:30:00Z"
  },
  "system_resources": {
    "memory_usage": "2.1 GB",
    "disk_usage": "45.2 GB",
    "cpu_usage": "15%"
  }
}
```

---

## WebSocket API (Future)

For real-time features, we plan to implement WebSocket endpoints:

### `/ws/mining`

Real-time mining progress updates.

### `/ws/search`

Real-time search result streaming.

---

## Rate Limiting

Current implementation does not enforce rate limiting. For production use, consider implementing:

- User-based rate limiting (e.g., 100 requests/hour)
- IP-based rate limiting (e.g., 1000 requests/hour)
- Endpoint-specific limits (e.g., 10 mining jobs/day)

---

## SDK Examples

### Python

```python
import requests

# Basic question answering
response = requests.post(
    "http://localhost:5000/ask",
    json={
        "question": "How to synthesize silver nanoparticles?",
        "mode": "protocol",
        "k_doc": 5
    }
)
result = response.json()
print(result["answer"])

# File upload with question
with open("protocol.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:5000/ask",
        data={"question": "Analyze this protocol"},
        files={"file": f}
    )
```

### JavaScript

```javascript
// Basic question answering
const response = await fetch('http://localhost:5000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'How to synthesize silver nanoparticles?',
    mode: 'protocol',
    k_doc: 5
  })
});

const result = await response.json();
console.log(result.answer);

// File upload
const formData = new FormData();
formData.append('question', 'Analyze this protocol');
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch('http://localhost:5000/ask', {
  method: 'POST',
  body: formData
});
```

### curl

```bash
# Basic question
curl -X POST "http://localhost:5000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How to synthesize silver nanoparticles?",
    "mode": "protocol",
    "k_doc": 5
  }'

# File upload
curl -X POST "http://localhost:5000/ask" \
  -F "question=Analyze this protocol" \
  -F "file=@protocol.pdf"

# Protocol conversion
curl -X POST "http://localhost:5000/convert" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Heat to 80°C and stir for 2 hours",
    "validate": true
  }'
```

---

This API documentation provides comprehensive coverage of all endpoints with examples and detailed parameter descriptions for publication-ready documentation.