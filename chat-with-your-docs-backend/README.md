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

| Method   | Path                       | Auth Required  | Description                                                          |
|----------|----------------------------|----------------|----------------------------------------------------------------------|
| `GET`    | `/`                        | Public         | API info, links to docs and state                                    |
| `POST`   | `/api/auth/signup`         | Public         | Create account `{name, email, password}` (201 with token & user)     |
| `POST`   | `/api/auth/login`          | Public         | Authenticate `{email, password}` (200 with token & user)             |
| `GET`    | `/api/auth/me`             | Bearer Token   | Return profile of currently authenticated user                       |
| `POST`   | `/api/auth/change-password`| Bearer Token   | Change password `{current_password, new_password}` (204)             |
| `POST`   | `/api/auth/logout`         | Public/Bearer  | Stateless logout signal (client discards bearer token, 204)          |
| `GET`    | `/api/admin/users`         | Admin Bearer   | List registered users with private file counts (admins only)         |
| `GET`    | `/api/state`               | Bearer Token   | User's private library files, pipeline readiness, and active models  |
| `POST`   | `/api/upload`              | Bearer Token   | Upload files into user's private library (`.pdf .docx .md .txt`)     |
| `DELETE` | `/api/files/{name}`        | Bearer Token   | Delete single file from user's library (404 for unknown/other users) |
| `DELETE` | `/api/files`               | Bearer Token   | Delete all files in user's private library                           |
| `POST`   | `/api/samples`             | Bearer Token   | Load sample notes into user's private library and index them         |
| `POST`   | `/api/chat`                | Bearer Token   | Stream RAG answer over user's private library as **NDJSON**          |

#### Upload rules

- Accepted extensions: `.pdf`, `.docx`, `.md`, `.txt`
- Max size: **50 MB** per file
- Filenames are sanitised (basename only, no `..`, safe characters only)
- Same filename replaces the previous file in the user's library
- Scanned / image-only PDFs are rejected with a clear error

---

## Authentication & User Isolation

### How it Works
- **Password Security**: Passwords are saved as Argon2id hashes using `argon2-cffi` (default parameters). Verification runs in constant time, including dummy verification for unknown emails to prevent timing-based user enumeration.
- **Access Tokens**: PyJWT issues signed HS256 tokens carrying `sub` (user ID), `role`, `iat`, `exp`, and unique `jti`.
- **Database**: Zero-ORM SQLite table `users` stored in `data/app.db` with WAL mode and parameterized queries.
- **Rate Limiting**: In-memory tracker keyed by `(client IP, lowercase email)` locks logins for `LOGIN_LOCK_MINUTES` after `LOGIN_MAX_FAILURES` consecutive bad attempts (returns HTTP 429 with `Retry-After`). Signups are limited to 10 per hour per IP.
- **Per-User Library Isolation**: Each user gets a dedicated directory under `data/library/<user_id>/` and an independent `UserLibrary` instance. Uploads and queries use a per-user lock, so users never block each other. An LRU registry keeps at most 5 active pipelines in memory simultaneously.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET` | Auto-generated | 64-byte secret. If unset, generated once and saved to `data/.jwt_secret`. **Set explicitly in production!** |
| `ACCESS_TOKEN_MINUTES` | `480` (8 hours) | Access token lifespan |
| `ADMIN_EMAILS` | `""` | Comma-separated list of emails assigned the `admin` role upon signup |
| `AUTH_DB_PATH` | `data/app.db` | Path to the SQLite user database |
| `MIN_PASSWORD_LENGTH` | `8` | Minimum password length (max 128) |
| `LOGIN_MAX_FAILURES` | `5` | Maximum failed logins before temporary lockout |
| `LOGIN_LOCK_MINUTES` | `15` | Lockout duration in minutes |

### PowerShell Walkthrough

```powershell
# 1. Sign up a new user account
curl.exe -X POST -H "Content-Type: application/json" `
  -d "{\"name\":\"Asha\",\"email\":\"asha@example.com\",\"password\":\"correct-horse-9\"}" `
  http://127.0.0.1:8000/api/auth/signup

# Example response:
# {"token":"<JWT_TOKEN>","user":{"id":"...","name":"Asha","email":"asha@example.com","role":"user"}}

# Store token in a variable
$TOKEN = "<JWT_TOKEN>"

# 2. Check user's private library state
curl.exe -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/state

# 3. Upload a document to private library
curl.exe -H "Authorization: Bearer $TOKEN" -F "files=@data\notes\01_intro_ml.md" http://127.0.0.1:8000/api/upload

# 4. Stream chat over private documents
curl.exe -N -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" `
  -d "{\"question\":\"What is machine learning?\",\"history\":[],\"mode\":\"fast\"}" `
  http://127.0.0.1:8000/api/chat

# 5. Check user profile
curl.exe -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/auth/me
```

### Test in Swagger UI (/docs)
1. Open **http://127.0.0.1:8000/docs** in your browser.
2. Sign up or log in via `POST /api/auth/login` or `POST /api/auth/signup` and copy the returned `token`.
3. Click the green **Authorize** button at the top right of the page.
4. Paste the token into the **Value** box and click **Authorize**.
5. All protected `/api/*` endpoints will now automatically send the `Authorization: Bearer <token>` header.

### Isolation & Migration Notes
- **Pre-existing files**: Documents sitting directly in `data/library/` from before this upgrade are left untouched and ignored. To migrate them into a user's library, move them into `data/library/<user_id>/` manually.
- **Shared Embedding Cache**: The vector cache (`index/embeddings.npz` or cache files) remains shared across users based on SHA-256 hashes of text chunks + embedding model to minimize disk usage and avoid redundant GPU/CPU re-computation. The cache file itself is global.

### Security Notes
- **HTTPS**: Required whenever the API is deployed or exposed beyond `localhost`.
- **Stateless Tokens**: JWTs are stateless; `POST /api/auth/logout` allows client-side token discard.
- **Rate Limiting**: Rate limits are stored in memory and reset if the process restarts.
- **Future Enhancements**: Email verification and self-serve password resets are not yet implemented.

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
