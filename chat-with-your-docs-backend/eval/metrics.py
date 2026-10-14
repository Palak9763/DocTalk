"""Pure evaluation metrics for retrieval benchmarking (no models, no network).

Provides hit@k, recall@k, precision@k, MRR, nDCG@k, retrieval reports,
and robust aggregation across evaluation items.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def hit_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    """Return 1.0 if at least one relevant document is in the top-k, else 0.0.

    Args:
        ranked: Ranked list of document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Cutoff rank (1-indexed).

    Returns:
        1.0 or 0.0.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = ranked[:k]
    for doc_id in top_k:
        if doc_id in relevant:
            return 1.0
    return 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    """Return fraction of relevant documents retrieved in the top-k.

    Args:
        ranked: Ranked list of document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if relevant is empty.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = set(ranked[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    """Return fraction of top-k retrieved documents that are relevant.

    Args:
        ranked: Ranked list of document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if k <= 0 or relevant is empty.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = set(ranked[:k])
    hits = len(top_k & relevant)
    return hits / k


def mrr(ranked: list[str], relevant: set[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for the first relevant document.

    Args:
        ranked: Ranked list of document IDs.
        relevant: Set of ground-truth relevant document IDs.

    Returns:
        1.0 / (rank + 1) for 0-indexed rank, or 0.0 if no relevant document found.
    """
    if not relevant or not ranked:
        return 0.0
    for idx, doc_id in enumerate(ranked):
        if doc_id in relevant:
            return 1.0 / (idx + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain with binary relevance.

    DCG@k = sum_{i=1}^k (rel_i / log2(i + 1))
    IDCG@k = sum_{i=1}^{min(k, |relevant|)} (1 / log2(i + 1))

    Args:
        ranked: Ranked list of document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        nDCG score in [0.0, 1.0].
    """
    if not relevant or not ranked or k <= 0:
        return 0.0

    top_k = ranked[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def retrieval_report(
    ranked: list[str], relevant: set[str], k: int = 5
) -> dict[str, float]:
    """Generate a dictionary report of all retrieval metrics at cutoff k.

    Args:
        ranked: Ranked list of retrieved document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Rank cutoff parameter (default 5).

    Returns:
        Dict mapping metric names (e.g. 'hit@5', 'recall@5', 'mrr') to float scores.
    """
    return {
        f"hit@{k}": hit_at_k(ranked, relevant, k),
        f"recall@{k}": recall_at_k(ranked, relevant, k),
        f"precision@{k}": precision_at_k(ranked, relevant, k),
        "mrr": mrr(ranked, relevant),
        f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
    }


def aggregate(
    reports: list[Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Aggregate a sequence of metric reports, skipping empty or invalid items.

    Args:
        reports: List of metric report dictionaries (or None for skipped items).
                 Items with empty relevant sets or marked '_skip': True are skipped.

    Returns:
        Dictionary containing mean scores per metric and the count of evaluated items.
    """
    valid_reports: list[Mapping[str, Any]] = []
    skipped_count = 0

    for rep in reports:
        if rep is None:
            skipped_count += 1
            continue
        if rep.get("_skip") is True or rep.get("is_empty_gold") is True:
            skipped_count += 1
            continue
        valid_reports.append(rep)

    if not valid_reports:
        return {
            "count": 0,
            "skipped": skipped_count,
        }

    # Identify all numeric metric keys across valid reports
    metric_keys = [
        k
        for k in valid_reports[0].keys()
        if not k.startswith("_") and isinstance(valid_reports[0][k], (int, float))
    ]

    result: dict[str, Any] = {
        "count": len(valid_reports),
        "skipped": skipped_count,
    }

    for key in metric_keys:
        values = [
            float(r[key])
            for r in valid_reports
            if key in r and isinstance(r[key], (int, float))
        ]
        result[key] = sum(values) / len(values) if values else 0.0

    return result
