"""Evaluation suite package for Chat With Your Docs."""

from eval.metrics import (
    aggregate,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    retrieval_report,
)
from eval.ragas_style import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    get_judge_stats,
    reset_judge_stats,
)

__all__ = [
    "hit_at_k",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "retrieval_report",
    "aggregate",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "get_judge_stats",
    "reset_judge_stats",
]
