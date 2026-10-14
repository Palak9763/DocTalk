# Phase 7: The Evaluation Suite

The evaluation suite proves, with quantitative numbers, the empirical contribution of each retrieval and generation technique across the RAG pipeline:
- **BM25 Lexical Search** (`sparse`)
- **Dense Bi-Encoder Vectors** (`dense`)
- **Hybrid Fusion with Reciprocal Rank Fusion** (`hybrid`)
- **Cross-Encoder Reranking** (`hybrid_rerank`)
- **Conversational Query Rewriting** (`+ rewrite`)
- **Multi-Perspective Query Expansion** (`+ expand`)
- **Hypothetical Document Embeddings** (`+ hyde`)
- **Self-RAG Routing, Grading & Citation Verification** (`full_selfrag`)

---

## 1. Quick Start

### Fast Retrieval Evaluation (No Answer Generation)
Evaluates Hit@5, Recall@5, Precision@5, MRR, and nDCG@5 across all 8 configurations:
```powershell
.\.venv\Scripts\python.exe -m eval.run_eval --mode retrieval
```

### Quick Sub-Experiment (Limit Questions & Select Configs)
```powershell
.\.venv\Scripts\python.exe -m eval.run_eval --mode retrieval --limit 5 --configs sparse,dense,hybrid,hybrid_rerank
```

### Full End-to-End Evaluation (With Answers & RAGAS Metrics)
Generates answers and evaluates context precision, context recall, faithfulness, and answer relevancy:
```powershell
.\.venv\Scripts\python.exe -m eval.run_eval --mode full --limit 6 --configs hybrid_rerank,full_selfrag
```

---

## 2. CLI Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | string | *required* | `retrieval` (fast, rank metrics) or `full` (answers + LLM judge) |
| `--docs` | path | `data/notes` | Directory containing source documents |
| `--dataset` | path | `eval/dataset.json` | Path to benchmark question dataset |
| `--configs` | string | `all 8 configs` | Comma-separated configs (e.g. `sparse,hybrid,full_selfrag`) |
| `--limit` | int | `None` (all) | Limit number of questions for quick iterations |
| `--judge` | string | `config.JUDGE_MODEL` | Judge LLM model name (defaults to `qwen2.5:3b-instruct`) |

Results are automatically saved to:
- `eval/results/<timestamp>_<mode>.json` (full raw per-question JSON records)
- `eval/results/<timestamp>_<mode>.md` (formatted markdown summary table)

---

## 3. Dataset Format (`eval/dataset.json`)

Each benchmark item adheres to the following JSON structure:

```json
{
  "id": "q01",
  "question": "Under what Gauss-Markov assumptions does OLS produce BLUE?",
  "history": [],
  "gold": [
    ["01_regression.md", "Linear Regression and Ordinary Least Squares"]
  ],
  "reference": "OLS produces the Best Linear Unbiased Estimator (BLUE) under linearity, homoscedasticity, no multicollinearity, and zero error expectation.",
  "type": "factual",
  "needs_retrieval": true
}
```

### Field Definitions:
- **`id`**: Unique question identifier (e.g. `q01`).
- **`question`**: The query text submitted to the pipeline.
- **`history`**: Conversational dialogue history: `[]` or `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`. Used to test conversational coreference rewriting in `follow_up` questions.
- **`gold`**: Array of `[filename, section_heading]` pairs that contain ground-truth context needed to answer the question. Must be `[]` for unanswerable or chit-chat queries.
- **`reference`**: Ground-truth answer extracted from the corpus. Empty `""` for unanswerable or chit-chat questions.
- **`type`**: Question category:
  - `factual`: Direct single-section retrieval questions.
  - `vocabulary_gap`: Queries phrased with terminology different from the notes (tests dense semantic search).
  - `exact_term`: Queries containing rare, specific mathematical formulas or acronyms (tests BM25 lexical recall).
  - `follow_up`: Multi-turn questions containing pronouns like "it" or "they" (tests query rewriting).
  - `multi_hop`: Involves concepts spanning two distinct document sections.
  - `no_retrieval`: Greetings or chit-chat (tests Self-RAG retrieval skipping).
  - `unanswerable`: Out-of-domain queries not covered in the corpus (tests hallucination avoidance).
- **`needs_retrieval`**: Boolean indicating whether retrieval is required (`true` for domain questions, `false` for greetings).

---

## 4. How to Create Gold Labels for Your Own Lecture PDF

When evaluating custom documents (e.g. `Cryptography.pdf`):

1. **Understand Document Sectioning**:
   - **Markdown (`.md`)**: Sections match markdown headings (e.g. `## Heading Text`).
   - **PDFs (`.pdf`)**: Each page is ingested as `Page 1`, `Page 2`, `Page 3`, etc.
   - **Word (`.docx`)**: Sections match `Heading 1`, `Heading 2`, etc.

2. **Inspect Ingested Chunks**:
   Run the following Python snippet to see the exact sections detected:
   ```powershell
   .\.venv\Scripts\python.exe -c "from rag.ingest import load_document; [print(c.source, '-->', c.section) for c in load_document('data/library/Cryptography.pdf')]"
   ```

3. **Format Gold Entries**:
   - For a PDF, point each relevant query to the page containing the answer:
     ```json
     "gold": [["Cryptography.pdf", "Page 2"]]
     ```
   - For a Markdown document:
     ```json
     "gold": [["02_neural_networks.md", "Activation Functions and the Dying ReLU Problem"]]
     ```
   - If a question requires synthesis from multiple pages:
     ```json
     "gold": [
       ["Cryptography.pdf", "Page 1"],
       ["Cryptography.pdf", "Page 3"]
     ]
     ```

4. **Run with Custom Document and Dataset**:
   ```powershell
   .\.venv\Scripts\python.exe -m eval.run_eval --mode retrieval --docs data/library --dataset my_custom_dataset.json
   ```

---

## 5. Metrics Overview

### Retrieval Metrics
- **Hit@5**: Binary indicator of whether at least one gold chunk appeared in the top 5 candidates.
- **Recall@5**: Proportion of all gold chunks retrieved within the top 5 candidates.
- **Precision@5**: Proportion of retrieved top-5 chunks that are ground-truth relevant.
- **MRR (Mean Reciprocal Rank)**: Reciprocal rank `1 / rank` of the first relevant retrieved document.
- **nDCG@5**: Normalized Discounted Cumulative Gain accounting for rank positions.

### RAGAS-Style LLM-Judged Metrics (0.0 to 1.0)
- **Context Precision**: Rank-weighted precision evaluating whether retrieved passages directly aid the question.
- **Context Recall**: Proportion of reference answer statements supported by the retrieved contexts.
- **Faithfulness**: Proportion of generated answer claims grounded directly in the retrieved contexts.
- **Answer Relevancy**: Cosine similarity between original query embedding and questions generated by the judge from the answer.
