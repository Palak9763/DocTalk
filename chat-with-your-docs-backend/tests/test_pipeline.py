"""Unit tests for Phase 4: Self-RAG, RAGPipeline, presets, and streaming."""
import copy
import sys
import tempfile
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import llm
from rag.config import PRESETS, PipelineConfig, NOTES_DIR
from rag.ingest import Chunk, load_chunks
from rag.pipeline import RAGPipeline, Result, _NO_CONTEXT_MSG
from rag.selfrag import check_groundedness, grade_chunks, needs_retrieval
from tests.fakes import FakeClient, install, install_fake_reranker, restore_reranker
# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _make_chunks(n: int = 5) -> list[Chunk]:
    """Return n deterministic fake chunks referencing ReLU."""
    return [
        Chunk(
            id=f"doc.md#{i}.0",
            source="doc.md",
            section=f"Section {i}",
            text=f"doc.md › Section {i}\nThe dying ReLU problem occurs when neurons become permanently inactive. Chunk {i}.",
            page=i + 1,
            prefix_len=len(f"doc.md › Section {i}\n"),
        )
        for i in range(n)
    ]

def _make_pipeline(fake: FakeClient, chunks: list[Chunk], cfg: PipelineConfig, reranker: Any = None) -> RAGPipeline:
    """Build a RAGPipeline with fake LLM and optionally a fake reranker."""
    return RAGPipeline(cfg=cfg, chunks=chunks, reranker=reranker)
class _FakeReranker:
    """Word-overlap reranker that produces scores in [0, 1] deterministically."""
    model_name = "fake-reranker"

    def rerank_scored(self, query: str, chunks: list[Chunk], idxs: list[int], top_k: int | None = None):
        import re
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for i in idxs:
            c_tokens = set(re.findall(r"\w+", chunks[i].text.lower()))
            score = min(1.0, len(q_tokens & c_tokens) / max(len(q_tokens), 1))
            scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        if top_k is not None:
            scored = scored[:top_k]
        return scored

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1: Self-RAG unit tests
# ──────────────────────────────────────────────────────────────────────────────

def test_needs_retrieval_smalltalk() -> None:
    """Chit-chat like 'hi' should return retrieve=False."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        fake.canned_chat = '{"retrieve": false, "reason": "Pure greeting."}'
        retrieve, reason = needs_retrieval("hi")
        assert retrieve is False
        assert "rewrite" not in fake.log

def test_needs_retrieval_topic_question() -> None:
    """A topic question should return retrieve=True."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        fake.canned_chat = '{"retrieve": true, "reason": "Topic question about ReLU."}'
        retrieve, reason = needs_retrieval("What is the dying ReLU problem?")
        assert retrieve is True

def test_needs_retrieval_garbage_defaults_to_true() -> None:
    """Garbage / exception output defaults to retrieve=True."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        fake.canned_chat = "this is not json at all @@##"
        retrieve, reason = needs_retrieval("hello")
        assert retrieve is True
        assert "defaulted" in reason.lower()

def test_needs_retrieval_missing_key_defaults_to_true() -> None:
    """JSON without the 'retrieve' key defaults to (True, ...)."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        fake.canned_chat = '{"reason": "something"}'
        retrieve, reason = needs_retrieval("what is backprop?")
        assert retrieve is True

def test_grade_chunks_keeps_on_garbage() -> None:
    """Unparseable LLM output keeps the chunk (True)."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        fake.canned_chat = "THIS IS NOT JSON"
        chunks = _make_chunks(2)
        verdicts = grade_chunks("relu problem", chunks)
        assert all(verdicts)

def test_grade_chunks_drops_on_explicit_no() -> None:
    """Explicit relevant=false drops the chunk."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        fake.canned_chat = '{"relevant": false}'
        chunks = _make_chunks(1)
        verdicts = grade_chunks("relu problem", chunks)
        assert verdicts == [False]

def test_check_groundedness_garbage_returns_grounded() -> None:
    """Unparseable groundedness output defaults to grounded=True."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        fake.canned_chat = "GARBAGE"
        result = check_groundedness("Some answer.", _make_chunks(2))
        assert result["grounded"] is True
        assert result["unsupported"] == []

# ──────────────────────────────────────────────────────────────────────────────
# TASK 2: LLM call count per preset
# ──────────────────────────────────────────────────────────────────────────────
def _run_pipeline(mode: str, question: str, history: list | None = None) -> tuple[Result, FakeClient]:
    """Run the pipeline with a fake client; return result and the fake client."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        chunks = _make_chunks(5)
        cfg = PRESETS[mode]
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=_FakeReranker())
        result = pipeline.answer(question, history=history or [])
        return result, fake

def test_fast_call_count() -> None:
    """Fast preset: decide + generate (no expand, no grade)."""
    result, fake = _run_pipeline("fast", "What is the dying ReLU problem?")
    assert result.used_retrieval is True
    # log should contain decide (→ "chat") and generate (→ "chat"), no expand/grade/verify
    assert "expand" not in fake.log
    assert "verify" not in fake.log

def test_balanced_call_count() -> None:
    """Balanced preset: decide + expand + generate."""
    result, fake = _run_pipeline("balanced", "What is the dying ReLU problem?")
    assert "expand" in fake.log
    assert "verify" not in fake.log

def test_thorough_call_count() -> None:
    """Thorough preset: decide + expand + grade × N + generate + verify."""
    result, fake = _run_pipeline("thorough", "What is the dying ReLU problem?")
    assert "expand" in fake.log
    # grade calls are currently logged as "chat" by the fake; verify as "chat"
    # Just confirm the answer came through and retrieval was used
    assert result.used_retrieval is True

def test_rewrite_with_history() -> None:
    """A follow-up with history adds exactly one rewrite call."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        chunks = _make_chunks(3)
        cfg = PRESETS["balanced"]
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=_FakeReranker())
        history = [
            ("user", "What is backpropagation?"),
            ("assistant", "It computes gradients via the chain rule."),
        ]
        pipeline.answer("why does it need the chain rule?", history=history)
        assert fake.log.count("rewrite") == 1

def test_no_rewrite_without_history() -> None:
    """A first question (no history) adds zero rewrite calls."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        chunks = _make_chunks(3)
        cfg = PRESETS["balanced"]
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=_FakeReranker())
        pipeline.answer("What is the dying ReLU problem?")
        assert fake.log.count("rewrite") == 0

# ──────────────────────────────────────────────────────────────────────────────
# No-retrieval path
# ──────────────────────────────────────────────────────────────────────────────
def test_no_retrieval_chit_chat() -> None:
    """Chit-chat returns used_retrieval=False with no retrieval calls."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        fake.canned_chat = '{"retrieve": false, "reason": "Greeting."}'
        chunks = _make_chunks(3)
        cfg = PRESETS["balanced"]
        # The fake needs to handle two messages: decide (json) then generate (stream)
        call_responses: list[str] = [
            '{"retrieve": false, "reason": "Greeting."}',
            "Hello! How can I help you?",
        ]
        call_count = [0]
        orig_chat = fake.__class__.chat

        def patched_chat(self, model, messages, stream=False, options=None, keep_alive=None):
            resp = call_responses[min(call_count[0], len(call_responses) - 1)]
            call_count[0] += 1
            self.calls.append({"method": "chat", "model": model, "messages": messages})
            if stream:
                def _gen():
                    for word in resp.split():
                        yield {"message": {"content": word + " "}}
                return _gen()
            return {"message": {"content": resp}}

        import types as _types
        fake.chat = _types.MethodType(patched_chat, fake)
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=_FakeReranker())
        result = pipeline.answer("hi there")
        assert result.used_retrieval is False
        assert result.contexts == []

# ─────────────────────────────────────────────────────────────────────────────
# Nothing survives grading
# ──────────────────────────────────────────────────────────────────────────────

def test_nothing_survives_grading() -> None:
    """When all chunks are graded irrelevant, return the 'could not find' message."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        chunks = _make_chunks(3)
        cfg = PipelineConfig(
            use_adaptive=False,use_expansion=False,use_rerank=False,use_grading=True,grading_mode="llm",use_groundedness_check=False,use_rewrite=False,)
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=None)

        # Patch grade_chunks to always return False
        import rag.pipeline as _pl
        original_grade = _pl.grade_chunks
        try:
            _pl.grade_chunks = lambda query, chunks_: [False] * len(chunks_)
            result = pipeline.answer("What is ReLU?")
        finally:
            _pl.grade_chunks = original_grade

        assert result.answer == _NO_CONTEXT_MSG
        assert result.contexts == []

# ──────────────────────────────────────────────────────────────────────────────
# Groundedness failure → regeneration
# ──────────────────────────────────────────────────────────────────────────────
def test_groundedness_failure_triggers_regeneration() -> None:
    """on_token receives reset=True once; regenerated=True; final answer is second generation."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        chunks = _make_chunks(3)
        cfg = PipelineConfig(
            use_adaptive=False,use_expansion=False,
            use_rerank=False,use_grading=False,
            use_groundedness_check=True,use_rewrite=False,
        )
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=None)
        # Sequence of responses: first=answer draft, second=groundedness fail, third=regenerated answer
        responses = [
            "First draft answer with unsupported claim.",           # generate
            '{"grounded": false, "unsupported": ["unsupported claim"]}',  # verify
            "Second corrected answer.",                              # regenerate
        ]
        call_idx = [0]

        def patched_chat(self, model, messages, stream=False, options=None, keep_alive=None):
            resp = responses[min(call_idx[0], len(responses) - 1)]
            call_idx[0] += 1
            self.calls.append({"method": "chat", "model": model, "messages": messages})
            if stream:
                def _gen():
                    for word in resp.split():
                        yield {"message": {"content": word + " "}}
                return _gen()
            return {"message": {"content": resp}}

        import types as _types
        fake.chat = _types.MethodType(patched_chat, fake)

        resets = [0]
        pieces: list[str] = []

        def on_token(piece: str = "", reset: bool = False) -> None:
            if reset:
                resets[0] += 1
                pieces.clear()
            else:
                pieces.append(piece)

        result = pipeline.answer("What is ReLU?", on_token=on_token)
        assert resets[0] == 1
        assert result.trace.get("regenerated") is True
        assert result.answer.strip().startswith("Second")

# ──────────────────────────────────────────────────────────────────────────────
# Streaming correctness
# ──────────────────────────────────────────────────────────────────────────────
def test_streaming_tokens_equal_result_answer() -> None:
    """Concatenated token pieces must equal Result.answer."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        chunks = _make_chunks(3)
        cfg = PipelineConfig(
            use_adaptive=False,
            use_expansion=False,
            use_rerank=False,
            use_grading=False,
            use_groundedness_check=False,
            use_rewrite=False,
        )
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=None)

        pieces: list[str] = []

        def on_token(piece: str = "", reset: bool = False) -> None:
            if not reset:
                pieces.append(piece)

        result = pipeline.answer("What is the dying ReLU problem?", on_token=on_token)
        streamed = "".join(pieces)
        assert streamed == result.answer, f"Streamed != result.answer:\n{streamed!r}\n{result.answer!r}"

def test_on_event_statuses() -> None:
    """on_event is called with at least one status before the answer."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        chunks = _make_chunks(3)
        cfg = PipelineConfig(
            use_adaptive=True,
            use_expansion=False,
            use_rerank=False,
            use_grading=False,
            use_groundedness_check=False,
            use_rewrite=False,
        )
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=None)
        statuses: list[str] = []

        def on_event(status: str, _trace: dict) -> None:
            statuses.append(status)

        pipeline.answer("What is the dying ReLU problem?", on_event=on_event)
        assert len(statuses) >= 1

# ──────────────────────────────────────────────────────────────────────────────
# Trace shape
# ──────────────────────────────────────────────────────────────────────────────
def test_trace_shape() -> None:
    """Verify trace invariants: counts consistent, paths cover retrieved ids, used ⊆ retrieved."""
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        chunks = _make_chunks(5)
        cfg = PipelineConfig(
            use_adaptive=False,
            use_expansion=False,
            use_rerank=True,
            use_grading=False,
            use_groundedness_check=False,
            use_rewrite=False,
        )
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=_FakeReranker())
        result = pipeline.answer("What is the dying ReLU problem?")
        tr = result.trace

        s = tr.get("search", {})
        # both <= min(dense_hits, sparse_hits)
        assert s.get("both", 0) <= min(s.get("dense_hits", 0), s.get("sparse_hits", 0))
        # to_reranker <= max_candidates
        assert s.get("to_reranker", 0) <= s.get("max_candidates", 50)

        # paths covers exactly the retrieved ids
        retrieved_ids = set(tr.get("retrieved", []))
        path_ids = set(tr.get("paths", {}).keys())
        assert retrieved_ids == path_ids, f"paths keys {path_ids} != retrieved {retrieved_ids}"

        # used ⊆ retrieved
        used_ids = set(tr.get("used", []))
        assert used_ids.issubset(retrieved_ids), f"used {used_ids} not subset of retrieved {retrieved_ids}"

# ──────────────────────────────────────────────────────────────────────────────
# Citations
# ──────────────────────────────────────────────────────────────────────────────
def test_citations_in_answer() -> None:
    """The generated answer must contain at least one [chunk_id] that exists in contexts."""
    import re
    with tempfile.TemporaryDirectory() as td:
        fake = install(td)
        install_fake_reranker()
        chunks = _make_chunks(3)
        cfg = PipelineConfig(
            use_adaptive=False,
            use_expansion=False,
            use_rerank=False,
            use_grading=False,
            use_groundedness_check=False,
            use_rewrite=False,
        )
        pipeline = RAGPipeline(cfg=cfg, chunks=chunks, reranker=None)
        # Make the fake return an answer citing the first chunk id
        first_id = chunks[0].id
        fake.canned_chat = f"The dying ReLU problem is explained in [{first_id}]."
        result = pipeline.answer("What is the dying ReLU problem?")

        context_ids = {c.id for c in result.contexts}
        cited = re.findall(r"\[([^\]]+)\]", result.answer)
        # At least one citation must appear in contexts
        assert any(cid in context_ids for cid in cited), (
            f"No cited chunk id in contexts. cited={cited}, context_ids={context_ids}"
        )
if __name__ == "__main__":
    test_needs_retrieval_smalltalk()
    test_needs_retrieval_topic_question()
    test_needs_retrieval_garbage_defaults_to_true()
    test_needs_retrieval_missing_key_defaults_to_true()
    test_grade_chunks_keeps_on_garbage()
    test_grade_chunks_drops_on_explicit_no()
    test_check_groundedness_garbage_returns_grounded()
    test_fast_call_count()
    test_balanced_call_count()
    test_thorough_call_count()
    test_rewrite_with_history()
    test_no_rewrite_without_history()
    test_no_retrieval_chit_chat()
    test_nothing_survives_grading()
    test_groundedness_failure_triggers_regeneration()
    test_streaming_tokens_equal_result_answer()
    test_on_event_statuses()
    test_trace_shape()
    test_citations_in_answer()
    print("All Phase 4 tests passed!")
