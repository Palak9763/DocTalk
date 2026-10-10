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

```bash
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
