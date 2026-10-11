"""Unit tests for Phase 3: Reranker and Query processing (rewrite, expansion, HyDE)."""

import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import NOTES_DIR
from rag.ingest import Chunk, load_chunks
from rag import llm
from rag.query import expand_query, hyde_passage, rewrite_query
from rag.retrieve import Reranker, Retriever
from tests.fakes import FakeClient, install, install_fake_reranker, restore_reranker


def test_reranker_lazy_load_and_fake_scorer() -> None:
    """Verify Reranker is lazy-loaded, respects fake scorer, sorts descending, and handles empty idxs."""
    reranker = Reranker()
    # Lazy load check: model is not yet loaded in memory
    assert reranker.loaded is False

    # Install fake scorer for testing without downloading or loading models
    install_fake_reranker()

    # Empty idxs returns []
    assert reranker.score("any query", [], []) == []
    assert reranker.rerank_scored("any query", [], [], top_k=5) == []
    assert reranker.rerank("any query", [], [], top_k=5) == []

    # Test scoring with sample chunks
    chunks = [
        Chunk(id="c0", source="notes.md", section="Backprop", text="backpropagation gradient chain rule calculation", page=1),
        Chunk(id="c1", source="notes.md", section="CNN", text="convolutional filters computer vision kernel stride", page=1),
        Chunk(id="c2", source="notes.md", section="Optimization", text="backpropagation optimization loss surface update", page=2),
    ]
    idxs = [0, 1, 2]

    scores = reranker.score("backpropagation gradient", chunks, idxs)
    assert len(scores) == 3
    for s in scores:
        assert 0.0 <= s <= 1.0

    scored = reranker.rerank_scored("backpropagation gradient", chunks, idxs, top_k=2)
    assert len(scored) == 2
    # Sorted descending by score
    assert scored[0][1] >= scored[1][1]
    # Chunk 0 has both 'backpropagation' and 'gradient', should be top
    assert scored[0][0] == 0

    top_idxs = reranker.rerank("backpropagation gradient", chunks, idxs, top_k=2)
    assert top_idxs == [scored[0][0], scored[1][0]]


def test_reranker_sigmoid_path() -> None:
    """Verify sigmoid transformation on raw logits outside [0, 1] preserves ranking."""
    restore_reranker()
    reranker = Reranker()

    class DummyCrossEncoder:
        def predict(self, pairs: list[Any], batch_size: int = 16, show_progress_bar: bool = False) -> np.ndarray:
            return np.array([-3.0, 0.0, 4.0])

    reranker.load = lambda: DummyCrossEncoder()  # type: ignore[method-assign]


    dummy_chunks = [
        Chunk(id="d0", source="doc.md", section="A", text="doc A", page=1),
        Chunk(id="d1", source="doc.md", section="B", text="doc B", page=1),
        Chunk(id="d2", source="doc.md", section="C", text="doc C", page=1),
    ]
    scores = reranker.score("test query", dummy_chunks, [0, 1, 2])
    assert len(scores) == 3

    # All transformed scores must be strictly in (0, 1) and preserve order
    for s in scores:
        assert 0.0 < s < 1.0
    assert scores[0] < scores[1] < scores[2]
    # Check exact sigmoid values
    assert math.isclose(scores[1], 0.5, abs_tol=1e-5)
    expected_neg3 = 1.0 / (1.0 + math.exp(3.0))
    expected_pos4 = 1.0 / (1.0 + math.exp(-4.0))
    assert math.isclose(scores[0], expected_neg3, rel_tol=1e-4)
    assert math.isclose(scores[2], expected_pos4, rel_tol=1e-4)


def test_rewrite_query(tmp_path: Path | None = None) -> None:
    """Verify rewrite_query behaviour on empty history, pronoun resolution, prefix/chatter stripping, and fallbacks."""
    with tempfile.TemporaryDirectory() as td:
        idx_dir = Path(tmp_path) if tmp_path is not None else Path(td)
        fake = install(idx_dir)

        # 1. Empty history returns question unchanged with ZERO LLM calls
        q1 = "Why does it need the chain rule?"
        res1 = rewrite_query(q1, history=[])
        assert res1 == q1
        assert len(fake.calls) == 0

        res1_none = rewrite_query(q1, history=None)
        assert res1_none == q1
        assert len(fake.calls) == 0

        # 2. History resolves pronoun
        history = [
            ("user", "What is backpropagation?"),
            ("assistant", "It computes gradients layer by layer using the chain rule."),
        ]
        res2 = rewrite_query("why does it need the chain rule?", history=history)
        assert "backpropagation" in res2.lower()
        assert "it" not in res2.lower().split()
        assert "rewrite" in fake.log

        # 3. Chatter and quotes stripped
        fake.canned_chat = 'Rewritten question: "Why does backpropagation need the chain rule?"\nNote: this is standalone.'
        res3 = rewrite_query("why does it need the chain rule?", history=history)
        assert res3 == "Why does backpropagation need the chain rule?"
        assert not res3.startswith('"') and not res3.endswith('"')

        # 4. Empty model output falls back to original question
        fake.canned_chat = ""
        res4 = rewrite_query("why does it need the chain rule?", history=history)
        assert res4 == "why does it need the chain rule?"

        # 5. Overlong model output (> 3x question length) falls back to original question
        fake.canned_chat = (
            "This is an extraordinarily long explanation that is more than three times the total character "
            "length of the original question and rambles on without stopping."
        )
        res5 = rewrite_query("why does it need?", history=history)
        assert res5 == "why does it need?"


def test_expand_query(tmp_path: Path | None = None) -> None:
    """Verify expand_query handles JSON objects, bare lists, fenced JSON, duplicates, echoes, and respects n."""
    with tempfile.TemporaryDirectory() as td:
        idx_dir = Path(tmp_path) if tmp_path is not None else Path(td)
        fake = install(idx_dir)

        # 1. Standard JSON object via fake client
        res1 = expand_query("backpropagation", n=2)
        assert len(res1) == 2
        assert all(isinstance(q, str) and len(q) > 0 for q in res1)
        assert "expand" in fake.log

        # 2. Bare list handling
        original_chat_json = llm.chat_json
        try:
            llm.chat_json = lambda *args, **kwargs: ["neural network training", "gradient descent"]  # type: ignore[assignment]
            res_list = expand_query("backpropagation", n=2)
            assert res_list == ["neural network training", "gradient descent"]

            # 3. Code-fenced JSON string parsed via parse_json inside chat_json
            fake.canned_chat = '```json\n{"queries": ["gradient calculation", "loss backprop"]}\n```'
            llm.chat_json = original_chat_json
            res_fence = expand_query("backpropagation", n=2)
            assert res_fence == ["gradient calculation", "loss backprop"]

            # 4. Duplicates (case-insensitive) dropped
            llm.chat_json = lambda *args, **kwargs: {"queries": ["Gradient Descent", "gradient descent", "SGD"]}  # type: ignore[assignment]
            res_dups = expand_query("backpropagation", n=3)
            assert res_dups == ["Gradient Descent", "SGD"]

            # 5. Echo of original query dropped
            llm.chat_json = lambda *args, **kwargs: {"queries": ["Backpropagation", "backpropagation", "loss propagation"]}  # type: ignore[assignment]
            res_echo = expand_query("backpropagation", n=2)
            assert res_echo == ["loss propagation"]

            # 6. Junk and empty items dropped, total failure returns []
            llm.chat_json = lambda *args, **kwargs: {"queries": [123, None, "", "   ", "valid query"]}  # type: ignore[assignment]
            res_junk = expand_query("backpropagation", n=2)
            assert res_junk == ["valid query"]

            llm.chat_json = lambda *args, **kwargs: {}  # type: ignore[assignment]
            assert expand_query("backpropagation", n=2) == []

            # 7. Respects n cap
            llm.chat_json = lambda *args, **kwargs: {"queries": ["q1", "q2", "q3", "q4"]}  # type: ignore[assignment]
            res_n = expand_query("backpropagation", n=2)
            assert len(res_n) == 2
            assert res_n == ["q1", "q2"]
        finally:
            llm.chat_json = original_chat_json


def test_hyde_passage(tmp_path: Path | None = None) -> None:
    """Verify hyde_passage returns hypothetical lecture text and returns empty string on failure."""
    with tempfile.TemporaryDirectory() as td:
        idx_dir = Path(tmp_path) if tmp_path is not None else Path(td)
        fake = install(idx_dir)

        passage = hyde_passage("What is backpropagation?")
        assert isinstance(passage, str)
        assert len(passage) > 0
        assert "hyde" in fake.log

        # Empty query returns ""
        assert hyde_passage("") == ""
        assert hyde_passage("   ") == ""

        # Failure returns ""
        fake.canned_chat = ""
        assert hyde_passage("any query") == ""


def test_integration_sample_corpus(tmp_path: Path | None = None) -> None:
    """Integration check: load notes, search for 'dying ReLU problem', and rerank top-5 with fake reranker."""
    with tempfile.TemporaryDirectory() as td:
        idx_dir = Path(tmp_path) if tmp_path is not None else Path(td)
        install(idx_dir)
        install_fake_reranker()

        chunks = load_chunks(NOTES_DIR)
        assert len(chunks) > 0, "Expected lecture notes in NOTES_DIR"

        retriever = Retriever(chunks, use_dense=True)
        query = "What is the dying ReLU problem?"
        search_res = retriever.search(query, mode="hybrid")
        candidates = search_res.candidates
        assert len(candidates) > 0, "Expected candidates from retriever search"

        reranker = Reranker()
        top_scored = reranker.rerank_scored(query, chunks, candidates, top_k=5)
        assert len(top_scored) <= 5, f"Expected at most 5 results, got {len(top_scored)}"
        assert len(top_scored) > 0, "Expected at least 1 reranked result"

        top_chunk = chunks[top_scored[0][0]]
        section_and_text = f"{top_chunk.section} {top_chunk.text}".lower()
        assert "relu" in section_and_text, f"Expected top chunk to mention ReLU, got section: {top_chunk.section}"


if __name__ == "__main__":
    test_reranker_lazy_load_and_fake_scorer()
    test_reranker_sigmoid_path()
    test_rewrite_query()
    test_expand_query()
    test_hyde_passage()
    test_integration_sample_corpus()
    print("All Phase 3 tests passed successfully!")
