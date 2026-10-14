"""RAG pipeline: joins Self-RAG routing, query rewriting/expansion/HyDE,
hybrid retrieval, cross-encoder reranking, chunk grading, generation, and groundedness checking."""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from rag import llm
from rag.config import (
    NOTES_DIR,
    RERANK_MODEL,
    PipelineConfig,
    PRESETS,
    MAX_ANSWER_TOKENS,
)
from rag.ingest import Chunk, load_chunks, load_document
from rag.query import expand_query, hyde_passage, rewrite_query
from rag.retrieve import Retriever, SearchResult, Reranker
from rag.selfrag import check_groundedness, grade_chunks, needs_retrieval

logger = logging.getLogger(__name__)

_NO_CONTEXT_MSG = (
    "I could not find anything relevant to that in your documents."
)


@dataclass
class Result:
    """Container for a completed pipeline run."""

    answer: str
    used_retrieval: bool
    contexts: list[Chunk]
    trace: dict[str, Any]


def _ms(start: float) -> int:
    """Return elapsed milliseconds since start."""
    return int((time.perf_counter() - start) * 1000)


def _dedup_queries(queries: list[str]) -> list[str]:
    """Return queries deduplicated case-insensitively, preserving original order."""
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(q)
    return result


def _format_history_for_llm(history: list[Any]) -> list[dict[str, str]]:
    """Convert history tuples/dicts to Ollama message format."""
    messages: list[dict[str, str]] = []
    for turn in history[-6:]:
        if isinstance(turn, (tuple, list)) and len(turn) >= 2:
            role = str(turn[0])
            content = str(turn[1])
        elif isinstance(turn, dict):
            role = turn.get("role", "user")
            content = turn.get("content", "")
        else:
            continue
        if role in ("user", "assistant", "system"):
            messages.append({"role": role, "content": content})
    return messages


class RAGPipeline:
    """Full end-to-end RAG pipeline joining all Phase 0-3 modules."""

    def __init__(
        self,
        cfg: PipelineConfig | None = None,
        chunks: list[Chunk] | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.cfg = cfg or PRESETS["balanced"]
        self.chunks = chunks if chunks is not None else load_chunks(NOTES_DIR)
        self.by_id: dict[str, Chunk] = {c.id: c for c in self.chunks}

        use_dense = self.cfg.retrieval in ("dense", "hybrid")
        self.retriever = Retriever(
            self.chunks,
            use_dense=use_dense,
            stem=self.cfg.use_stemming,
        )

        # Reranker: accept injected one (tests), build lazily if cfg asks for it
        if reranker is not None:
            self._reranker: Reranker | None = reranker
        elif self.cfg.use_rerank:
            self._reranker = Reranker(RERANK_MODEL)  # lazy – no load at construction
        else:
            self._reranker = None

    def with_config(self, cfg: PipelineConfig) -> "RAGPipeline":
        """Return a shallow copy sharing index and models but with a different config."""
        clone = copy.copy(self)
        clone.cfg = cfg
        if cfg.use_rerank and clone._reranker is None:
            clone._reranker = Reranker(RERANK_MODEL)
        elif not cfg.use_rerank:
            clone._reranker = None
        return clone

    # ------------------------------------------------------------------
    # Internal Stage B + C + D
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        history: list[Any] | None = None,
        trace: dict[str, Any] | None = None,
        emit: Callable[[str, dict], None] | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """Run stages B (rewrite/expand/HyDE), C (search), D (rerank).

        Returns (candidate_indices, updated_trace).
        """
        cfg = self.cfg
        if trace is None:
            trace = {}
        timings: dict[str, int] = trace.setdefault("timings_ms", {})

        # ── Stage B: Query rewriting ──
        original_q = question
        rewritten_q = question
        trace["original"] = original_q

        if cfg.use_rewrite and history:
            if emit:
                emit("Rewriting query…", copy.deepcopy(trace))
            t0 = time.perf_counter()
            rewritten_q = rewrite_query(question, history=history)
            timings["rewrite"] = _ms(t0)
        trace["query"] = rewritten_q

        # ── Stage B: Expansion ──
        extra_queries: list[str] = []
        if cfg.use_expansion:
            if emit:
                emit("Expanding query…", copy.deepcopy(trace))
            t0 = time.perf_counter()
            extra_queries = expand_query(rewritten_q, n=cfg.n_expansions)
            timings["expand"] = _ms(t0)

        all_queries = _dedup_queries([rewritten_q] + extra_queries)
        trace["queries"] = all_queries

        # ── Stage B: HyDE ──
        hyde_texts: list[str] | None = None
        if cfg.use_hyde:
            if emit:
                emit("Generating HyDE passage…", copy.deepcopy(trace))
            passage = hyde_passage(rewritten_q)
            if passage:
                hyde_texts = [passage]
                trace["hyde"] = passage

        # ── Stage C: Retrieval ──
        if emit:
            emit("Searching corpus…", copy.deepcopy(trace))
        t0 = time.perf_counter()
        search_res: SearchResult = self.retriever.search(
            queries=all_queries,
            mode=cfg.retrieval,
            dense_k=cfg.dense_k,
            sparse_k=cfg.sparse_k,
            rrf_k=cfg.rrf_k,
            max_candidates=cfg.max_candidates,
            hyde_texts=hyde_texts,
            dedupe=cfg.dedupe_content,
        )
        timings["search"] = _ms(t0)
        candidates = search_res.candidates
        trace["candidates"] = len(candidates)
        trace["retrieval_mode"] = cfg.retrieval

        # Compute dense/sparse/both hit counts for trace
        dense_hits = set(search_res.dense_rank.keys())
        sparse_hits = set(search_res.sparse_rank.keys())
        both_hits = dense_hits & sparse_hits
        trace["search"] = {
            "dense_k": cfg.dense_k,
            "sparse_k": cfg.sparse_k,
            "rrf_k": cfg.rrf_k,
            "max_candidates": cfg.max_candidates,
            "dense_hits": len(dense_hits),
            "sparse_hits": len(sparse_hits),
            "both": len(both_hits),
            "fused": search_res.fused_total,
            "duplicates_removed": search_res.duplicates_removed,
            "to_reranker": min(len(candidates), cfg.rerank_top_n),
            "dense_index": (
                "hnsw" if (
                    self.retriever.dense_index is not None
                    and hasattr(self.retriever.dense_index, "_index")
                    and self.retriever.dense_index._index is not None
                ) else (
                    "exact" if self.retriever.dense_index is not None else None
                )
            ),
            "stemming": cfg.use_stemming,
        }

        # ── Stage D: Reranking ──
        rerank_scores: dict[str, float] = {}
        if self._reranker is not None and candidates:
            if emit:
                emit("Reranking candidates…", copy.deepcopy(trace))
            pool = candidates[: cfg.rerank_top_n]
            t0 = time.perf_counter()
            scored = self._reranker.rerank_scored(rewritten_q, self.chunks, pool, top_k=None)
            timings["rerank"] = _ms(t0)
            # Re-sort and take final_k
            scored.sort(key=lambda x: -x[1])
            top_idxs = [idx for idx, _ in scored[: cfg.final_k]]
            for idx, sc in scored:
                cid = self.chunks[idx].id
                rerank_scores[cid] = round(sc, 3)
            trace["reranker"] = self._reranker.model_name
        else:
            # No reranker – take first final_k candidates
            top_idxs = candidates[: cfg.final_k]
            trace["reranker"] = None

        trace["rerank_scores"] = rerank_scores
        trace["retrieved"] = [self.chunks[i].id for i in top_idxs]

        # Path provenance (dense rank / sparse rank per retrieved chunk)
        paths: dict[str, dict[str, int | None]] = {}
        for idx in top_idxs:
            cid = self.chunks[idx].id
            paths[cid] = {
                "dense": search_res.dense_rank.get(idx),
                "sparse": search_res.sparse_rank.get(idx),
            }
        trace["paths"] = paths

        return top_idxs, trace

    # ------------------------------------------------------------------
    # Public: answer()
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        history: list[Any] | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        on_token: Callable[..., None] | None = None,
    ) -> Result:
        """Run the full pipeline and return a Result.

        Args:
            question: User question.
            history: List of (role, content) tuples / dicts.
            on_event: Called with (status_text, trace_copy) at each stage start.
            on_token: Called with (piece, reset=False) for streamed tokens;
                      on_token("", reset=True) signals the UI to discard draft.

        Returns:
            Result dataclass.
        """
        cfg = self.cfg
        history = history or []
        trace: dict[str, Any] = {
            "question": question,
            "config": cfg.label(),
            "timings_ms": {},
        }
        timings: dict[str, int] = trace["timings_ms"]
        total_start = time.perf_counter()

        def emit(status: str, tr: dict) -> None:
            if on_event:
                on_event(status, tr)

        # ── Stage A: Adaptive routing ──
        if cfg.use_adaptive:
            emit("Deciding whether retrieval is needed…", copy.deepcopy(trace))
            t0 = time.perf_counter()
            do_retrieve, reason = needs_retrieval(question, history)
            timings["decide"] = _ms(t0)
            trace["decision"] = {
                "enabled": True,
                "retrieve": do_retrieve,
                "reason": reason,
            }
            emit(f"Decision: {'retrieve' if do_retrieve else 'direct answer'}", copy.deepcopy(trace))
        else:
            do_retrieve = True
            trace["decision"] = {"enabled": False, "retrieve": True, "reason": "disabled"}

        # ── No retrieval path ──
        if not do_retrieve:
            emit("Generating direct answer…", copy.deepcopy(trace))
            sys_msg = (
                "You are a friendly, helpful assistant. "
                "Answer the user's message naturally and concisely."
            )
            msgs: list[dict[str, str]] = [{"role": "system", "content": sys_msg}]
            msgs.extend(_format_history_for_llm(history))
            msgs.append({"role": "user", "content": question})

            answer_parts: list[str] = []
            t0 = time.perf_counter()
            for piece in llm.chat_stream(msgs, temperature=0.3, max_tokens=MAX_ANSWER_TOKENS):
                answer_parts.append(piece)
                if on_token:
                    on_token(piece)
            timings["generate"] = _ms(t0)
            timings["total"] = _ms(total_start)
            final_answer = "".join(answer_parts)
            trace["used"] = []
            return Result(
                answer=final_answer,
                used_retrieval=False,
                contexts=[],
                trace=trace,
            )

        # ── Retrieval path (B + C + D) ──
        top_idxs, trace = self.retrieve(question, history, trace, emit)
        retrieved_chunks = [self.chunks[i] for i in top_idxs]

        # ── Stage E: Chunk Grading ──
        if cfg.use_grading and retrieved_chunks:
            emit("Grading chunk relevance…", copy.deepcopy(trace))
            graded_out: list[str] = []

            if cfg.grading_mode == "rerank" and trace.get("rerank_scores"):
                # Fast path: threshold by rerank score
                kept: list[Chunk] = []
                for c in retrieved_chunks:
                    score = trace["rerank_scores"].get(c.id, 0.0)
                    if score >= cfg.rerank_min_score:
                        kept.append(c)
                    else:
                        graded_out.append(c.id)
                trace["grading"] = "reranker"
            else:
                # LLM grading: one call per chunk
                t0 = time.perf_counter()
                verdicts = grade_chunks(question, retrieved_chunks)
                timings["grade"] = _ms(t0)
                kept = [c for c, keep in zip(retrieved_chunks, verdicts) if keep]
                graded_out = [c.id for c, keep in zip(retrieved_chunks, verdicts) if not keep]
                trace["grading"] = "llm"

            trace["graded_out"] = graded_out
            retrieved_chunks = kept

        # ── Nothing survived grading ──
        if not retrieved_chunks:
            timings["total"] = _ms(total_start)
            trace["used"] = []
            return Result(
                answer=_NO_CONTEXT_MSG,
                used_retrieval=True,
                contexts=[],
                trace=trace,
            )

        # ── Stage F: Answer generation ──
        emit("Generating answer…", copy.deepcopy(trace))
        used_ids = [c.id for c in retrieved_chunks]
        trace["used"] = used_ids

        excerpts = "\n\n".join(
            f"[{c.id}] ({c.source})\n{c.body[:800]}" for c in retrieved_chunks
        )
        sys_prompt = (
            "You are a document Q&A assistant. Answer strictly from the provided excerpts. "
            "Cite the source in square brackets after each claim, e.g. [report.pdf#2.0]. "
            "If the excerpts do not contain the answer, say so. Be concise."
        )
        history_msgs = _format_history_for_llm(history)
        gen_msgs: list[dict[str, str]] = [{"role": "system", "content": sys_prompt}]
        gen_msgs.extend(history_msgs)
        gen_msgs.append({
            "role": "user",
            "content": (
                f"Excerpts:\n{excerpts}\n\n"
                f"Question: {question}"
            ),
        })

        answer_parts = []
        t0 = time.perf_counter()
        for piece in llm.chat_stream(gen_msgs, temperature=0.0, max_tokens=MAX_ANSWER_TOKENS):
            answer_parts.append(piece)
            if on_token:
                on_token(piece)
        timings["generate"] = _ms(t0)
        first_answer = "".join(answer_parts)

        # ── Stage G: Groundedness check ──
        final_answer = first_answer
        if cfg.use_groundedness_check:
            emit("Checking groundedness…", copy.deepcopy(trace))
            t0 = time.perf_counter()
            ground_result = check_groundedness(first_answer, retrieved_chunks)
            timings["verify"] = _ms(t0)
            trace["groundedness"] = ground_result

            if not ground_result["grounded"]:
                # Reset stream and regenerate once
                if on_token:
                    on_token("", reset=True)
                unsupported_str = "; ".join(ground_result["unsupported"])
                regen_note = (
                    f"\n\nIMPORTANT: The following claims from your previous answer were NOT "
                    f"supported by the excerpts and must NOT be repeated: {unsupported_str}"
                )
                regen_msgs = list(gen_msgs)
                regen_msgs[-1] = {
                    "role": "user",
                    "content": gen_msgs[-1]["content"] + regen_note,
                }
                emit("Regenerating grounded answer…", copy.deepcopy(trace))
                answer_parts = []
                t0 = time.perf_counter()
                for piece in llm.chat_stream(regen_msgs, temperature=0.0, max_tokens=MAX_ANSWER_TOKENS):
                    answer_parts.append(piece)
                    if on_token:
                        on_token(piece)
                timings["generate"] += _ms(t0)
                final_answer = "".join(answer_parts)
                trace["regenerated"] = True

        timings["total"] = _ms(total_start)
        return Result(
            answer=final_answer,
            used_retrieval=True,
            contexts=retrieved_chunks,
            trace=trace,
        )
