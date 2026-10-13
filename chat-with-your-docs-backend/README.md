# Chat With Your Docs (Backend)

Chat With Your Docs is a production-grade local Retrieval-Augmented Generation (RAG) system built for lecture notes.
Designed for privacy-first, zero-cloud execution, the entire inference and retrieval pipeline runs locally on your machine.
This repository hosts the backend service (`chat-with-your-docs-backend`), developed in modular phases starting with Phase 0 (Foundation).
The architecture provides full ablation flexibility, allowing researchers and developers to evaluate every pipeline component independently.
FastAPI powers the REST endpoints while Ollama and sentence-transformers drive language modeling, dense embeddings, and cross-encoder reranking.
BM25 sparse search is implemented natively with Snowball stemming to capture exact lexical matches alongside dense semantic vectors.
Reciprocal Rank Fusion (RRF) harmonizes multi-source candidate pools before heavy cross-encoder scoring.
Self-RAG adaptive classification routes queries dynamically to conserve computational resources on direct requests.
Granular chunk grading and citation verification guarantee that answers remain strictly grounded in source lecture documents.
Configurable presets (`fast`, `balanced`, and `thorough`) provide turn-key profiles tailored to different hardware capabilities and latency budgets.

## Architecture

- **[A] Self-RAG**: Decides whether retrieval is required or if the query can be answered directly.
- **[B] Query Rewrite & Expansion**: Resolves conversational coreferences and generates multi-perspective search queries.
- **[C] Hybrid Retrieval**: Dense embeddings + BM25 inverted index, fused with Reciprocal Rank Fusion (RRF), deduplicated to top-50.
- **[D] Cross-Encoder Rerank**: Deep cross-attention re-scoring with `BAAI/bge-reranker-v2-m3`.
- **[E] Chunk Grading**: Relevance verification to prune noise chunks before generation (score threshold or LLM reflection).
- **[F] Answer Generation**: Grounded generation with citation markers using `qwen2.5:3b-instruct`.
- **[G] Groundedness Check**: Post-generation citation validation against retrieved source context.

## Local Stack

- **LLM**: `qwen2.5:3b-instruct` via Ollama
- **Embeddings**: `nomic-embed-text` via Ollama
- **Reranker**: `BAAI/bge-reranker-v2-m3` via sentence-transformers
- **Dense Vector Search**: `hnswlib` (HNSW)
- **Sparse Lexical Search**: Custom BM25 with `snowballstemmer`
- **Backend API**: `FastAPI` + `uvicorn`

## Setup

```powershell
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text
.\.venv\Scripts\python.exe -m venv .venv
pip install -r requirements.txt
```

---

## API (Phase 5)

### Starting the server

```powershell
.\.venv\Scripts\python.exe server.py
# Custom host/port
.\.venv\Scripts\python.exe server.py --host 0.0.0.0 --port 8080
```

Interactive Swagger docs are available at **http://127.0.0.1:8000/docs** once the server is running.

### Endpoint table

| Method   | Path                   | Description                                                          |
|----------|------------------------|----------------------------------------------------------------------|
| `GET`    | `/`                    | API info, links to docs and state                                    |
| `GET`    | `/api/state`           | Library files, pipeline readiness, model names, available modes      |
| `POST`   | `/api/upload`          | Upload one or more `.pdf .docx .md .txt` files (multipart `files`)   |
| `DELETE` | `/api/files/{name}`    | Delete a single file by name (404 for unknown or path-trick names)   |
| `DELETE` | `/api/files`           | Delete all uploaded files                                            |
| `POST`   | `/api/samples`         | Copy built-in sample lecture notes into the library and index them   |
| `POST`   | `/api/chat`            | Stream an answer as **NDJSON** (see below)                           |

#### Upload rules

- Accepted extensions: `.pdf`, `.docx`, `.md`, `.txt`
- Max size: **50 MB** per file
- Filenames are sanitised (basename only, no `..`, safe characters only)
- Same filename replaces the previous file (no duplicates)
- Scanned / image-only PDFs are rejected with a clear error

### POST /api/chat request body

```json
{
  "question": "What is backpropagation?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "mode": "balanced"
}
```

| Field      | Type   | Constraints                          | Default      |
|------------|--------|--------------------------------------|--------------|
| `question` | string | 1–2000 characters                    | *(required)* |
| `history`  | array  | max 20 `{role, content}` items       | `[]`         |
| `mode`     | string | `"fast"` \| `"balanced"` \| `"thorough"` | `"balanced"` |

Returns **HTTP 409** when no pipeline is ready. Returns **HTTP 422** for an unknown mode.

### NDJSON event stream format

`POST /api/chat` streams **one JSON object per line** (`application/x-ndjson`).  
Use `Cache-Control: no-cache` and parse lines incrementally.

| Event type | Payload fields                                                      | When emitted                              |
|------------|---------------------------------------------------------------------|-------------------------------------------|
| `trace`    | `status` (str), `trace` (dict)                                      | At every pipeline stage start             |
| `sources`  | `sources` (list), `terms` (list of highlight words)                 | Once retrieval + grading finish, **before** any token |
| `token`    | `text` (str)                                                        | Each streamed answer piece                |
| `reset`    | *(none)*                                                            | Discard draft (groundedness regeneration) |
| `done`     | `answer`, `used_retrieval`, `sources`, `terms`, `trace`             | Final – stream ends after this            |
| `error`    | `message` (str)                                                     | On failure; stream ends                   |

Each `source` object:
```json
{
  "id": "lecture.pdf#0.1",
  "source": "lecture.pdf",
  "section": "Page 1",
  "page": 1,
  "text": "Backpropagation computes gradients via the chain rule…",
  "score": 0.87,
  "dense_rank": 1,
  "sparse_rank": 3,
  "used": true
}
```

#### Example stream (fast mode)

```
{"type":"trace","status":"Deciding whether retrieval is needed…","trace":{…}}
{"type":"trace","status":"Decision: retrieve","trace":{…}}
{"type":"trace","status":"Searching corpus…","trace":{…}}
{"type":"sources","sources":[{"id":"lecture.pdf#0.0","source":"lecture.pdf","page":1,…}],"terms":["backpropagation"]}
{"type":"trace","status":"Generating answer…","trace":{…}}
{"type":"token","text":"Backpropagation "}
{"type":"token","text":"computes "}
{"type":"token","text":"gradients "}
{"type":"done","answer":"Backpropagation computes gradients …","used_retrieval":true,"sources":[…],"terms":["backpropagation"],"trace":{"timings_ms":{"decide":120,"search":340,"generate":980,"total":1480}}}
```

### CORS

By default the API allows cross-origin requests from:
`http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`, `http://localhost:8080`, `http://localhost:5500` (and `127.0.0.1` variants).

Override with the **`CORS_ORIGINS`** environment variable (comma-separated):

```powershell
$env:CORS_ORIGINS = "http://localhost:5173,https://my-frontend.example.com"
.\.venv\Scripts\python.exe server.py
```

### Test it without a frontend

**Option A – Swagger UI**

1. Open **http://127.0.0.1:8000/docs**
2. Use `POST /api/upload` → "Try it out" → pick a lecture PDF
3. Use `POST /api/chat` → "Try it out" → paste `{"question": "…", "history": [], "mode": "fast"}`

**Option B – PowerShell curl**

```powershell
# Upload a PDF
curl.exe -F "files=@C:\path\lecture.pdf" http://127.0.0.1:8000/api/upload

# Ask a question (streamed NDJSON, -N disables output buffering)
curl.exe -N -H "Content-Type: application/json" `
  -d "{\"question\":\"What is backpropagation?\",\"history\":[],\"mode\":\"fast\"}" `
  http://127.0.0.1:8000/api/chat

# Check state
curl.exe http://127.0.0.1:8000/api/state

# Delete a file
curl.exe -X DELETE http://127.0.0.1:8000/api/files/lecture.pdf
```

### Running tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
