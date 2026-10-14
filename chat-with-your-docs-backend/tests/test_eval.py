"""Unit tests for Phase 7 Evaluation Suite (pure fakes, no real models)."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from eval import metrics, ragas_style
from eval.run_eval import (
    CONFIG_SPECS,
    count_llm_calls,
    resolve_gold_chunk_ids,
    run_evaluation,
)
from rag import config
from rag.config import PipelineConfig
from rag.ingest import load_chunks
from tests import fakes


# ---------------------------------------------------------------------------
# 1. Hand-computed Metric Tests
# ---------------------------------------------------------------------------


def test_metrics_exact_hand_computed() -> None:
    """Verify exact values on known hand-computed ranking examples."""
    ranked = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d2", "d4"}

    # Top-1: ["d1"] -> hits: 0
    assert metrics.hit_at_k(ranked, relevant, k=1) == 0.0
    assert metrics.recall_at_k(ranked, relevant, k=1) == 0.0
    assert metrics.precision_at_k(ranked, relevant, k=1) == 0.0

    # Top-2: ["d1", "d2"] -> hits: 1 ("d2")
    assert metrics.hit_at_k(ranked, relevant, k=2) == 1.0
    assert metrics.recall_at_k(ranked, relevant, k=2) == 0.5
    assert metrics.precision_at_k(ranked, relevant, k=2) == 0.5

    # Top-4: ["d1", "d2", "d3", "d4"] -> hits: 2 ("d2", "d4")
    assert metrics.hit_at_k(ranked, relevant, k=4) == 1.0
    assert metrics.recall_at_k(ranked, relevant, k=4) == 1.0
    assert metrics.precision_at_k(ranked, relevant, k=4) == 0.5  # 2 / 4 = 0.5

    # MRR: first relevant is d2 at index 1 -> 1 / (1 + 1) = 0.5
    assert metrics.mrr(ranked, relevant) == 0.5

    # nDCG@3: ["d1", "d2", "d3"]
    # DCG@3 = 0/log2(2) + 1/log2(3) + 0/log2(4) = 1 / log2(3)
    # IDCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 1 / log2(3)
    expected_dcg = 1.0 / math.log2(3)
    expected_idcg = 1.0 + 1.0 / math.log2(3)
    expected_ndcg = expected_dcg / expected_idcg
    assert math.isclose(metrics.ndcg_at_k(ranked, relevant, k=3), expected_ndcg, rel_tol=1e-5)


def test_metrics_empty_and_edge_cases() -> None:
    """Verify edge cases: empty ranked, empty relevant, k larger than list."""
    assert metrics.hit_at_k([], {"d1"}, k=5) == 0.0
    assert metrics.recall_at_k([], {"d1"}, k=5) == 0.0
    assert metrics.precision_at_k([], {"d1"}, k=5) == 0.0
    assert metrics.mrr([], {"d1"}) == 0.0
    assert metrics.ndcg_at_k([], {"d1"}, k=5) == 0.0

    # Empty relevant set
    assert metrics.hit_at_k(["d1", "d2"], set(), k=5) == 0.0
    assert metrics.recall_at_k(["d1", "d2"], set(), k=5) == 0.0
    assert metrics.precision_at_k(["d1", "d2"], set(), k=5) == 0.0
    assert metrics.mrr(["d1", "d2"], set()) == 0.0
    assert metrics.ndcg_at_k(["d1", "d2"], set(), k=5) == 0.0

    # k larger than list
    ranked = ["d1"]
    relevant = {"d1"}
    assert metrics.hit_at_k(ranked, relevant, k=10) == 1.0
    assert metrics.recall_at_k(ranked, relevant, k=10) == 1.0
    assert metrics.precision_at_k(ranked, relevant, k=10) == 0.1  # 1 / 10
    assert metrics.mrr(ranked, relevant) == 1.0
    assert metrics.ndcg_at_k(ranked, relevant, k=10) == 1.0


def test_aggregate_and_report() -> None:
    """Verify retrieval_report and aggregate skipping mechanics."""
    rep1 = metrics.retrieval_report(["d1"], {"d1"}, k=1)
    rep2 = metrics.retrieval_report(["d2"], {"d1"}, k=1)
    skip_rep = {"_skip": True, "is_empty_gold": True}

    agg = metrics.aggregate([rep1, rep2, None, skip_rep])
    assert agg["count"] == 2
    assert agg["skipped"] == 2
    assert agg["hit@1"] == 0.5  # (1.0 + 0.0) / 2
    assert agg["recall@1"] == 0.5
    assert agg["mrr"] == 0.5


# ---------------------------------------------------------------------------
# 2. Dataset Schema & Integrity Tests
# ---------------------------------------------------------------------------


def test_dataset_json_integrity() -> None:
    """Verify dataset.json resolves to real chunks, unique IDs, and covers all types."""
    ds_path = config.ROOT / "eval" / "dataset.json"
    assert ds_path.is_file(), "dataset.json does not exist"

    dataset: list[dict[str, Any]] = json.loads(ds_path.read_text(encoding="utf-8"))
    assert len(dataset) >= 20, f"Expected ~20 questions, got {len(dataset)}"

    # Check ID uniqueness
    ids = [item["id"] for item in dataset]
    assert len(ids) == len(set(ids)), "Question IDs must be unique"

    # Verify every required type is present at least twice
    expected_types = {
        "factual",
        "vocabulary_gap",
        "exact_term",
        "follow_up",
        "multi_hop",
        "no_retrieval",
        "unanswerable",
    }
    type_counts: dict[str, int] = {}
    for item in dataset:
        t = item.get("type")
        type_counts[t] = type_counts.get(t, 0) + 1

    for exp_t in expected_types:
        assert (
            type_counts.get(exp_t, 0) >= 2
        ), f"Question type '{exp_t}' must appear at least twice, found {type_counts.get(exp_t, 0)}"

    # Verify resolution to real chunks in data/notes
    chunks = load_chunks(config.NOTES_DIR)
    assert len(chunks) > 0

    chunk_map = {(c.source, c.section) for c in chunks}
    for item in dataset:
        gold = item.get("gold", [])
        if item["type"] in ("no_retrieval", "unanswerable"):
            assert gold == [], f"{item['id']} ({item['type']}) must have empty gold"
            assert item["reference"] == ""
        else:
            assert len(gold) >= 1, f"{item['id']} must specify at least one gold section"
            for src, sec in gold:
                assert (
                    src,
                    sec,
                ) in chunk_map, f"{item['id']} gold [{src}, {sec}] does not exist in notes chunks"


# ---------------------------------------------------------------------------
# 3. Ragas-Style Judge Metric Tests with Fakes
# ---------------------------------------------------------------------------


def test_ragas_style_with_fakes(tmp_path: Path) -> None:
    """Verify ragas_style metrics produce [0, 1] with mock judge responses."""
    fake = fakes.install(tmp_path)
    ragas_style.reset_judge_stats()

    # Configure fake judge to return well-formed evaluation responses
    fake.canned_chat = json.dumps({
        "useful": [True, False, True],
        "supported": [True, True],
        "questions": ["What is backpropagation?", "How does reverse-mode differentiation work?", "Why is chain rule used?"],
    })

    q = "What is backpropagation?"
    contexts = [
        "Backpropagation applies the calculus chain rule.",
        "A random unrelated passage about baking bread.",
        "Upstream gradient signals multiply local Jacobian matrices.",
    ]

    # Context precision
    cp = ragas_style.context_precision(q, contexts)
    assert cp is not None
    assert 0.0 <= cp <= 1.0

    # Context recall
    cr = ragas_style.context_recall(q, "Statement one. Statement two.", contexts)
    assert cr is not None
    assert 0.0 <= cr <= 1.0

    # Faithfulness
    faith = ragas_style.faithfulness(q, "Claim one. Claim two.", contexts)
    assert faith is not None
    assert 0.0 <= faith <= 1.0

    # Answer relevancy
    ans_rel = ragas_style.answer_relevancy(q, "Backpropagation calculates gradient vectors.")
    assert ans_rel is not None
    assert 0.0 <= ans_rel <= 1.0

    stats = ragas_style.get_judge_stats()
    assert stats["calls"] >= 4
    assert stats["skips"] == 0


def test_ragas_style_garbage_judge_output(tmp_path: Path) -> None:
    """Verify that unparseable 3B judge responses increment skip counter without crashing."""
    fake = fakes.install(tmp_path)
    ragas_style.reset_judge_stats()

    # Set non-JSON unparseable garbage output
    fake.canned_chat = "I am a helpful AI and I think this question is great! Here are thoughts..."

    cp = ragas_style.context_precision("Question?", ["Context passage."])
    assert cp is None

    cr = ragas_style.context_recall("Question?", "Reference statement.", ["Context passage."])
    assert cr is None

    faith = ragas_style.faithfulness("Question?", "Answer statement.", ["Context passage."])
    assert faith is None

    ans_rel = ragas_style.answer_relevancy("Question?", "Answer statement.")
    assert ans_rel is None

    stats = ragas_style.get_judge_stats()
    assert stats["calls"] >= 4
    assert stats["skips"] >= 4


# ---------------------------------------------------------------------------
# 4. run_eval Offline Integration Test
# ---------------------------------------------------------------------------


def test_run_eval_retrieval_mode_offline(tmp_path: Path) -> None:
    """Run run_eval on a limited subset with offline fake models and verify outputs."""
    fake = fakes.install(tmp_path)
    fakes.install_fake_reranker()

    # Create a small dataset with 3 valid items and 1 intentionally broken item
    test_ds = [
        {
            "id": "t1",
            "question": "What is backpropagation?",
            "history": [],
            "gold": [["02_neural_networks.md", "Backpropagation and the Computational Graph"]],
            "reference": "Calculus chain rule backward.",
            "type": "factual",
            "needs_retrieval": True,
        },
        {
            "id": "t2",
            "question": "Hello there!",
            "history": [],
            "gold": [],
            "reference": "",
            "type": "no_retrieval",
            "needs_retrieval": False,
        },
        {
            "id": "t3_broken",
            "question": None,  # Causes pipeline exception across configs to verify error capture
            "history": [],
            "gold": [["02_neural_networks.md", "Backpropagation and the Computational Graph"]],
            "reference": "Testing error capture.",
            "type": "factual",
            "needs_retrieval": True,
        },
    ]

    ds_file = tmp_path / "test_dataset.json"
    ds_file.write_text(json.dumps(test_ds), encoding="utf-8")

    # Run retrieval evaluation over selected configs
    configs_to_test = ["sparse", "dense", "hybrid", "hybrid_rerank"]
    raw_data, report_md = run_evaluation(
        mode="retrieval",
        docs_path=config.NOTES_DIR,
        dataset_path=ds_file,
        selected_configs=configs_to_test,
        limit=None,
    )

    # 1. Configs are distinct and present in output
    for c in configs_to_test:
        assert c in raw_data["config_summaries"]
        assert f"`{c}`" in report_md

    # 2. Check report files were created
    timestamp = raw_data["metadata"]["timestamp"]
    res_dir = config.ROOT / "eval" / "results"
    assert (res_dir / f"{timestamp}_retrieval.json").is_file()
    assert (res_dir / f"{timestamp}_retrieval.md").is_file()

    # 3. Broken question did not stop the run and was recorded
    for c in configs_to_test:
        results = raw_data["raw_results"][c]
        assert len(results) == 3
        # t3_broken was captured with error flag
        broken_res = next(r for r in results if r["id"] == "t3_broken")
        assert broken_res.get("_skip") is True

    # 4. Clean up mock reranker
    fakes.restore_reranker()


def test_helper_llm_call_count() -> None:
    """Verify count_llm_calls calculation across different pipeline configurations."""
    cfg = PipelineConfig(use_adaptive=True, use_rewrite=True, use_expansion=True)
    trace = {"timings_ms": {"decide": 50, "rewrite": 100, "expand": 150, "generate": 200}}
    assert count_llm_calls(cfg, trace, has_history=True) >= 4
