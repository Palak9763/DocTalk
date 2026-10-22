# DocTalk 📚

### Production-Grade Adaptive Retrieval-Augmented Generation (RAG) System

DocTalk is an intelligent document question-answering system that lets users upload documents and ask questions in natural language. It combines hybrid retrieval, query transformation, cross-encoder reranking, adaptive retrieval, and grounded answer generation to provide relevant responses with source citations.

Instead of sending entire documents to the language model, DocTalk retrieves the most relevant text chunks and uses them to generate context-aware answers.

## ✨ Features

- **Document Q&A:** Ask questions about your uploaded documents.
- **Multiple File Formats:** Supports PDF, DOCX, Markdown, and TXT files.
- **Adaptive Retrieval (Self-RAG):** Determines whether document retrieval is needed.
- **Query Rewriting:** Resolves follow-up questions using conversation history.
- **Query Expansion:** Generates alternative query phrasings to improve retrieval.
- **Hybrid Search:** Combines dense vector retrieval with sparse BM25 search.
- **Reciprocal Rank Fusion (RRF):** Merges rankings from multiple retrieval methods.
- **Cross-Encoder Reranking:** Reorders retrieved chunks by relevance using a BGE reranker.
- **Grounded Generation:** Generates answers using retrieved document context.
- **Source Citations:** Connects answers to the supporting document chunks.
- **Streaming Responses:** Displays generated answers progressively.
- **RAG Evaluation:** Evaluates retrieval and answer quality using metrics such as Recall@K, MRR, nDCG, and RAGAS-style measures.
- **Local Model Support:** Uses Ollama to run language and embedding models locally.

## 🏗️ Architecture

```text
                  User Query
                      |
                      v
             Adaptive Router
                (Self-RAG)
                      |
             Retrieval Required?
                /         \
              No           Yes
              |             |
        Direct Answer   Query Rewrite
                        & Expansion
                             |
                             v
                    Hybrid Retrieval
                  +---------------+
                  | Dense Search  |
                  | BM25 Search   |
                  +---------------+
                             |
                             v
                       RRF Fusion
                             |
                             v
                     BGE Reranker
                             |
                             v
                  Relevant Text Chunks
                             |
                             v
                  Grounded Generation
                             |
                             v
                  Answer Verification
                             |
                             v
                  Answer + Citations
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Backend API | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Language Model | Qwen 2.5 3B Instruct |
| Embedding Model | Nomic Embed Text |
| Reranker | BGE Cross-Encoder |
| Vector Search | HNSWLib |
| Sparse Retrieval | BM25 |
| Rank Fusion | Reciprocal Rank Fusion |
| Model Runtime | Ollama |
| Document Processing | PyPDF, python-docx |
| Evaluation | Custom retrieval metrics and RAGAS-style evaluation |

## 📁 Project Structure

```text
DocTalk/
├── chat-with-your-docs-backend/
│   ├── server.py
│   ├── rag/
│   ├── eval/
│   ├── data/
│   ├── index/
│   ├── requirements.txt
│   └── .env.example
│
├── chat-with-your-docs-frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── README.md
│
├── .gitignore
└── README.md
```

*Note: Update the folder and module names above if your repository uses a different structure.*

## ⚙️ Prerequisites

Before running DocTalk, install:

- Python 3.11 or later
- [Ollama](https://ollama.com/)
- Git
- A modern web browser

Pull the required local models:

```bash
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text
```

The BGE reranker is loaded through `sentence-transformers` and may download its model files the first time it runs.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Palak9763/DocTalk.git
cd DocTalk
```

### 2. Set up the backend

Open a terminal in the backend directory:

```bash
cd chat-with-your-docs-backend
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file using your project's supported configuration options. For example:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Set model names and paths according to the actual variables supported by `rag/config.py`. Do not commit secrets or machine-specific configuration.

### 4. Start the backend

Make sure Ollama is running, then start the API:

```bash
python server.py
```

The API is expected to run at:

```text
http://127.0.0.1:8000
```

If enabled, interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the frontend

Open a second terminal in the frontend directory:

```bash
cd chat-with-your-docs-frontend
python -m http.server 5500
```

Open the application in your browser:

```text
http://127.0.0.1:5500
```

Ensure that the frontend API URL matches the address where the backend is running.

## 💬 How It Works

1. **Upload:** The user uploads a supported document.
2. **Parse and chunk:** Text is extracted and split into smaller overlapping chunks.
3. **Index:** The system builds sparse BM25 and dense vector indexes.
4. **Route:** The adaptive router determines whether document retrieval is necessary.
5. **Transform:** Follow-up questions are rewritten and alternative queries are generated when appropriate.
6. **Retrieve:** Dense and sparse retrieval run in parallel or sequence, depending on the implementation.
7. **Fuse:** Reciprocal Rank Fusion combines the candidate rankings.
8. **Rerank:** A BGE cross-encoder selects the most relevant chunks.
9. **Generate:** Qwen produces an answer grounded in the retrieved context.
10. **Cite and verify:** The system provides source references and, where configured, checks the answer against the evidence.

The language model receives retrieved text rather than the entire original PDF. Consequently, targeted questions work best; whole-document summaries may require a dedicated map-reduce or hierarchical summarization workflow.

## 📊 Evaluation

DocTalk includes an evaluation workflow for comparing different RAG configurations.

Potential evaluation metrics include:

| Metric | Purpose |
|---|---|
| Hit Rate@K | Checks whether a relevant chunk appears in the top K results |
| Recall@K | Measures how many relevant chunks are retrieved |
| Precision@K | Measures the proportion of retrieved chunks that are relevant |
| MRR | Measures the rank of the first relevant result |
| nDCG | Evaluates ranking quality while accounting for relevance |
| Faithfulness | Checks whether answers are supported by the supplied context |
| Answer Relevance | Evaluates whether answers address the question |
| Latency | Measures retrieval and response time |

Run the evaluation script provided in the backend:

```bash
python eval/run_eval.py
```

The exact command and available metrics depend on the evaluation script included in your repository.

## 🔒 Privacy and Security

- Uploaded documents are processed by the backend.
- Local Ollama models can keep inference on your machine.
- Do not expose the API publicly without appropriate authentication and security controls.
- Keep `.env`, private documents, embedding caches, and local indexes out of version control.
- Review any model-download or telemetry behavior associated with your installed dependencies.

## 🗺️ Future Improvements

- Adaptive retrieval retries with a bounded retry limit
- Explicit relevance and evidence-sufficiency grading
- Stronger answer verification and unsupported-claim handling
- HyDE and multi-step question decomposition
- OCR support for scanned PDFs
- Persistent vector database support
- Authentication and multi-user document isolation
- Docker-based deployment
- Expanded benchmark datasets and reproducible experiments

## 🎯 Project Goal

DocTalk demonstrates how a basic RAG application can be extended into an adaptive document intelligence system through hybrid retrieval, reranking, query transformation, grounded generation, and systematic evaluation.

## 👩‍💻 Author

**Palak Deshmukh**

GitHub: [@Palak9763](https://github.com/Palak9763)

---

*DocTalk — Talk to your documents.*
