# REQUIREMENTS_REPORT.md
## Chat With Your Docs — Production-Grade RAG System

**Project:** `chat-with-your-docs-backend`  
**Models:** LLM `qwen2.5:3b-instruct` · Embed `nomic-embed-text` · Reranker `BAAI/bge-reranker-v2-m3`  
**Corpus:** 16 chunks from 4 lecture-note markdown files (`data/notes/`)  
**Evaluation dataset:** 30 questions (`eval/dataset.json`) — 5 factual, 5 vocab-gap, 5 exact-term, 5 follow-up, 4 multi-hop, 3 no-retrieval, 3 unanswerable  
**Report generated:** 2026-10-30

---

## Compliance Summary

| Req | Status | Key Evidence |
|-----|:------:|--------------|
| **R1** Query Rewriting & Expansion | ✅ **Met** | `demo_requirements_live.txt` · `rag/query.py` |
| **R2** Hybrid Retrieval + Reranking | ✅ **Met** | `20261028_195209_retrieval.md` · `rag/retrieve.py` |
| **R3** Self-RAG / Adaptive Retrieval | ✅ **Met** | `demo_requirements_live.txt` · `rag/selfrag.py` |
| **R4** Formal Eval Suite (RAGAS-style) | ✅ **Met** | `eval/` directory · `20261028_195209_retrieval.md` · `20261014_150107_full.md` |

---

## Section 1 — Audit

### R1: Query Rewriting and Expansion

**Implementation files:** [`rag/query.py`](rag/query.py)  
**Key functions:** `rewrite_query`, `expand_query`, `hyde_passage`

| Behaviour | Verified |
|-----------|:--------:|
| `rewrite_query` is **skipped** when `history` is empty (line-level guard: `if not history: return query`) | ✅ |
| `rewrite_query` resolves coreferences when history is present (LLM prompt asks for standalone phrasing) | ✅ |
| `expand_query` returns **deduplicated** variants (case-insensitive set, original excluded) | ✅ |
| HyDE is **optional** (`use_hyde` flag, off by default in `fast` and `balanced` presets) | ✅ |
| All variants (rewritten + expansions + HyDE) flow into **both** dense and sparse retrieval paths via `all_queries` list in `Retriever.search()` | ✅ |

**PipelineConfig switches:**

| Switch | `fast` | `balanced` | `thorough` |
|--------|:------:|:----------:|:----------:|
| `use_rewrite` | `False` | `True` | `True` |
| `use_expansion` | `False` | `True` | `True` |
| `n_expansions` | 2 | 2 | 3 |
| `use_hyde` | `False` | `False` | `True` |

**Gaps found:** None.

---

### R2: Hybrid Dense + Sparse Retrieval, Plus Reranking

**Implementation files:** [`rag/retrieve.py`](rag/retrieve.py)  
**Key classes/functions:** `InvertedIndex`, `DenseIndex`, `Retriever.search()`, `rrf()`, `Reranker`

| Component | Implementation | Verified |
|-----------|---------------|:--------:|
| **Dense path** | `DenseIndex` — cosine similarity (exact) / HNSW (≥20 k chunks) via `nomic-embed-text` embeddings | ✅ |
| **Sparse path** | `InvertedIndex` — BM25 with Lucene IDF, Snowball stemmer, ~60-word STOPWORDS frozenset | ✅ |
| **RRF fusion** | `rrf(k=60)` combines both ranked lists | ✅ |
| **Deduplication** | MD5 fingerprint dedup after fusion | ✅ |
| **Top-50 cap** | `max_candidates=50` before reranker | ✅ |
| **Cross-encoder rerank** | `Reranker` — `BAAI/bge-reranker-v2-m3`, sigmoid-normalised scores, lazy-loaded, cached | ✅ |
| **Final top-5** | `final_k=5` (configurable) | ✅ |
| **Per-path ranks in trace** | `trace["paths"]` dict exposes `dense_rank` and `sparse_rank` per chunk | ✅ |

**PipelineConfig switches:**

| Switch | `fast` | `balanced` | `thorough` |
|--------|:------:|:----------:|:----------:|
| `retrieval` | `"hybrid"` | `"hybrid"` | `"hybrid"` |
| `dense_k` | 30 | 30 | 30 |
| `sparse_k` | 30 | 30 | 30 |
| `rrf_k` | 60 | 60 | 60 |
| `max_candidates` | 50 | 50 | 50 |
| `use_rerank` | `False` | `True` | `True` |
| `final_k` | 5 | 5 | 5 |

**Gaps found:** None.

---

### R3: Self-RAG / Adaptive Retrieval

**Implementation files:** [`rag/selfrag.py`](rag/selfrag.py), [`rag/pipeline.py`](rag/pipeline.py)  
**Key functions:** `needs_retrieval`, `grade_chunks`, `check_groundedness`

| Behaviour | Verified |
|-----------|:--------:|
| `needs_retrieval` runs **before** any retrieval (pipeline Stage A) | ✅ |
| Small-talk / greetings route to `retrieve=False`, zero sources returned | ✅ |
| Default on LLM parse failure is **True** (retrieve on doubt) | ✅ |
| `grade_chunks` supports two modes: `reranker-score` threshold and `llm` per-chunk grading | ✅ |
| Chunks with insufficient support are filtered; if all removed → `_NO_CONTEXT_MSG` returned | ✅ |
| `check_groundedness` fires after generation; if `grounded=False` → one regeneration loop | ✅ |

**PipelineConfig switches:**

| Switch | `fast` | `balanced` | `thorough` |
|--------|:------:|:----------:|:----------:|
| `use_adaptive` | `False` | `True` | `True` |
| `use_grading` | `False` | `True` | `True` |
| `grading_mode` | `"reranker-score"` | `"reranker-score"` | `"llm"` |
| `rerank_min_score` | 0.0 | 0.1 | 0.1 |
| `use_groundedness_check` | `False` | `False` | `True` |

**Gaps found:** None.

---

### R4: Formal Retrieval-Quality Evaluation Suite

**Implementation files:** [`eval/metrics.py`](eval/metrics.py), [`eval/ragas_style.py`](eval/ragas_style.py), [`eval/run_eval.py`](eval/run_eval.py), [`eval/dataset.json`](eval/dataset.json), [`eval/README.md`](eval/README.md)

| Component | Details | Verified |
|-----------|---------|:--------:|
| **Dataset** | 30 questions, all types with gold chunk IDs verified against real chunks | ✅ |
| **Retrieval metrics** | Hit@5, Recall@5, Precision@5, MRR, nDCG@5 | ✅ |
| **RAGAS-style metrics** | Context Precision, Context Recall, Faithfulness, Answer Relevancy (LLM judge) | ✅ |
| **Bootstrap CIs** | 1,000 resamples, 95% percentile CI on paired metric deltas | ✅ |
| **Ablation configs** | 8 cumulative configs (sparse → full_selfrag) | ✅ |
| **By-type breakdowns** | factual, vocab-gap, exact-term, follow-up, multi-hop | ✅ |
| **All tests pass** | `pytest tests/` → 84 passed | ✅ |

**Gaps found:** None.

---

## Section 2 — Key Numbers from Real Runs

> All figures are drawn directly from files in `eval/results/`. No numbers are invented.

### Run 1 — `demo_requirements.py --live` (R1/R2/R3 proof)
**Output file:** `eval/results/demo_requirements_live.txt`

| Check | Result |
|-------|--------|
| R1: Rewritten query differs from original | **PASS** — `"why does it need the chain rule?"` → `"Why does backpropagation need the chain rule?"` |
| R1: Expansion variants are unique | **PASS** — 2 unique variants (case-insensitive dedup confirmed) |
| R1: HyDE passage generated | **PASS** — passage produced when `use_hyde=True` |
| R2: Both dense and sparse paths contributed | **PASS** — `dense_hits=16`, `sparse_hits=6`, `found_by_both=6` |
| R2: Rerank order differs from fusion order | **PASS** — printed both orders, confirmed difference |
| R2: Per-path ranks exposed in trace | **PASS** — `trace["paths"]` shows `dense_rank`/`sparse_rank` per chunk |
| R3: `"hello"` → `retrieve=False`, 0 sources | **PASS** |
| R3: Topic question → retrieves | **PASS** |
| R3: Unanswerable → chunks graded out | **PASS** — 5 chunks removed, `_NO_CONTEXT_MSG` returned |
| R3: Groundedness check fires and regenerates | **PASS** — `groundedness=False`, `regenerated=True` |

**Exit code:** 0 (all checks passed)

---

### Run 2 — Retrieval Ablation (30 questions, 8 configs)
**Output file:** `eval/results/20261028_195209_retrieval.md`  
**Date:** 2026-10-28 · Corpus: 16 chunks · Questions evaluated: 24 (6 skipped: no_retrieval/unanswerable)

#### Cumulative Ablation Table

| Config | Pipeline Toggles | MRR | nDCG@5 | Recall@5 |
|:-------|:-----------------|:---:|:------:|:--------:|
| `sparse` | BM25 (Snowball stemming, Lucene IDF) | **1.000** | 0.995 | 1.000 |
| `dense` | Bi-encoder cosine (nomic-embed-text) | 0.899 | 0.910 | 1.000 |
| `hybrid` | + RRF fusion (k=60) | 0.958 | 0.969 | 1.000 |
| `hybrid_rerank` | + cross-encoder rerank (bge-reranker-v2-m3) | **1.000** | **0.997** | 1.000 |
| `+ rewrite` | + query rewriting | **1.000** | **0.997** | 1.000 |
| `+ expand` | + query expansion (2 variants) | **1.000** | **0.997** | 1.000 |
| `+ hyde` | + HyDE passages | **1.000** | **0.997** | 1.000 |
| `full_selfrag` | + Self-RAG adaptive routing + grading | **1.000** | **0.997** | 1.000 |

#### Bootstrap CIs (95%, 1,000 resamples)

| Comparison | Δ MRR [95% CI] | Interpretation |
|-----------|:--------------:|----------------|
| `dense` vs `sparse` | −0.101 [−0.201, −0.021] | **Significant** — BM25 outperforms dense on this small corpus |
| `hybrid` vs `dense` | +0.059 [0.000, +0.132] | Marginal improvement from RRF |
| `hybrid_rerank` vs `hybrid` | +0.042 [0.000, +0.104] | Reranker adds ranking precision |
| `+ rewrite` vs `hybrid_rerank` | +0.000 [0.000, 0.000] | No marginal gain (already at ceiling) |

#### By Question Type (selected configs)

| Type | `sparse` MRR | `dense` MRR | `hybrid_rerank` MRR | Key Insight |
|------|:------------:|:-----------:|:-------------------:|-------------|
| `factual` | 1.000 | 1.000 | 1.000 | All configs perfect |
| `exact_term` | 1.000 | 1.000 | 1.000 | BM25 lexical match trivially wins |
| `follow_up` | 1.000 | **0.717** | **1.000** | Dense most hurt by coreference; reranker restores |
| `vocabulary_gap` | 1.000 | **0.800** | **1.000** | Dense penalised by OOV; reranker restores |
| `multi_hop` | 1.000 | 1.000 | 1.000 | All configs perfect (small corpus) |

---

### Run 3 — Full Eval (RAGAS-style, 3 questions, 8 configs)
**Output file:** `eval/results/20261014_150107_full.md`  
**Date:** 2026-10-14 · Judge calls: 93 · Judge skips: 1

#### End-to-End Metrics (avg over 3 factual questions)

| Config | Avg Latency | LLM Calls/q | Faithfulness | Answer Relevancy | Skip Acc |
|:-------|:-----------:|:-----------:|:------------:|:----------------:|:--------:|
| `sparse` | 20,131 ms | 1.0 | 0.500 | 0.549 | 1.00 |
| `dense` | 17,987 ms | 1.0 | 0.500 | 0.561 | 1.00 |
| `hybrid` | 12,147 ms | 1.0 | 0.500 | 0.561 | 1.00 |
| `hybrid_rerank` | 60,790 ms | 1.0 | 0.167 | 0.561 | 1.00 |
| `+ rewrite` | 32,730 ms | 1.0 | 0.167 | 0.561 | 1.00 |
| `+ expand` | 35,466 ms | 2.0 | 0.167 | 0.561 | 1.00 |
| `+ hyde` | 39,847 ms | 3.0 | 0.167 | 0.561 | 1.00 |
| `full_selfrag` | **62,615 ms** | **8.7** | **0.500** | **0.625** | 1.00 |

> **Note:** Faithfulness and context precision scores are noisy — the 3B judge model produces high variance on a 3-question sample. The `full_selfrag` answer relevancy improvement (+0.064 vs sparse) is directionally consistent but not statistically significant at this sample size.

**Decision accuracy:** 1.00 for all configs (all 3 questions correctly routed to retrieval).  
**Latency cost of Self-RAG:** ~62.6 s/q vs ~12.1 s/q for bare hybrid (+417%) due to 8.7 LLM calls per question (grading + groundedness).

---

## Section 3 — Limitations

1. **Corpus size (16 chunks):** Near-ceiling Recall@5 and Hit@5 for every config. MRR is the only metric that differentiates configurations. A larger corpus would reveal more meaningful gaps.

2. **Dataset size (30 questions):** Bootstrap CIs are wide (many include 0), confirming that improvements beyond `hybrid_rerank` cannot be claimed as statistically significant on this dataset alone.

3. **3B judge model noise:** `qwen2.5:3b-instruct` used as RAGAS judge produces inconsistent faithfulness scores (e.g. 0.167 vs 0.500 across configs with identical retrieval). A larger judge model or human annotation would give cleaner results.

4. **BM25 beats dense (corpus artifact):** BM25 outperforms the bi-encoder baseline (MRR 1.000 vs 0.899) because the corpus is small and the questions are carefully phrased to match lecture-note vocabulary — this advantage would likely reverse on a larger, more diverse corpus.

5. **No OCR support:** Scanned PDFs cannot be ingested. Only text-extractable PDFs, DOCX, MD, and TXT are supported.

6. **Single-node, local-only:** Ollama models and the BM25 index are loaded in memory with no distributed serving layer.

---

## Section 4 — Reproduction Commands

```powershell
# Activate virtual environment
cd d:\DocTalk\chat-with-your-docs-backend

# Step 1: Run all tests (expect 84 passed)
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Step 2: R1/R2/R3 demo (requires Ollama running)
.\.venv\Scripts\python.exe demo_requirements.py --live 2>&1 | Tee-Object -FilePath eval\results\demo_requirements_live.txt

# Step 3: Retrieval ablation (30 questions, 8 configs, ~5 min)
.\.venv\Scripts\python.exe -m eval.run_eval --mode retrieval

# Step 4: Full RAGAS eval (LLM judge, slower — use --limit to cap)
.\.venv\Scripts\python.exe -m eval.run_eval --mode full --limit 10 --configs hybrid_rerank,full_selfrag

# Step 5: Fake (offline) demo — no Ollama needed
.\.venv\Scripts\python.exe demo_requirements.py
```

**Results are written to:** `eval/results/<timestamp>_<mode>.md` and `.json`

---

## Section 5 — File Inventory

| File | Purpose |
|------|---------|
| `rag/query.py` | R1: `rewrite_query`, `expand_query`, `hyde_passage` |
| `rag/retrieve.py` | R2: `InvertedIndex` (BM25), `DenseIndex`, `Retriever`, `rrf`, `Reranker` |
| `rag/selfrag.py` | R3: `needs_retrieval`, `grade_chunks`, `check_groundedness` |
| `rag/pipeline.py` | Orchestrates all stages; exposes `trace` dict |
| `rag/config.py` | `PipelineConfig` with `fast`/`balanced`/`thorough` presets |
| `rag/ingest.py` | Document loading, chunking, embedding |
| `rag/llm.py` | Ollama LLM wrapper |
| `eval/dataset.json` | 30 evaluation questions with gold chunk IDs |
| `eval/metrics.py` | Hit@5, Recall@5, MRR, nDCG@5, bootstrap CI |
| `eval/ragas_style.py` | Context precision/recall, faithfulness, answer relevancy |
| `eval/run_eval.py` | CLI entry point for all eval modes |
| `eval/README.md` | Eval suite documentation |
| `demo_requirements.py` | R1–R3 proof demo (fake + live modes) |
| `tests/test_demo_requirements.py` | Automated demo tests (fake mode) |
| `tests/test_eval.py` | Eval suite unit tests |
| `eval/results/20261028_195209_retrieval.md` | **Primary retrieval evidence** (30q, 8 configs) |
| `eval/results/20261014_150107_full.md` | **Full RAGAS evidence** (3q, 8 configs, 93 judge calls) |
| `eval/results/demo_requirements_live.txt` | **R1/R2/R3 live demo output** |

