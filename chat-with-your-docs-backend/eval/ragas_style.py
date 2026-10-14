"""LLM-judged RAGAS-style evaluation metrics using local judge models.

Implements four core metrics with tolerance for messy 3B-model JSON:
- context_precision: Rank-weighted precision of retrieved contexts
- context_recall: Fraction of reference statements supported by context
- faithfulness: Fraction of generated answer claims grounded in context
- answer_relevancy: Mean cosine similarity of generated questions to original query
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from rag import config, llm

logger = logging.getLogger(__name__)

# Global tracking for judge LLM consumption
_judge_calls: int = 0
_judge_skips: int = 0


def get_judge_stats() -> dict[str, int]:
    """Return total judge model calls made and skipped malformed items."""
    return {"calls": _judge_calls, "skips": _judge_skips}


def reset_judge_stats() -> None:
    """Reset judge call and skip counters to zero."""
    global _judge_calls, _judge_skips
    _judge_calls = 0
    _judge_skips = 0


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, stripping whitespace and citation markers."""
    if not text:
        return []
    # Remove citation chips like [1], [2] before splitting
    cleaned = re.sub(r"\[\d+\]", "", text).strip()
    raw = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in raw if len(s.strip()) > 3]


def _call_judge(
    messages: list[dict[str, str]],
    max_tokens: int = 150,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Invoke the judge model via chat_json and track execution counts."""
    global _judge_calls
    _judge_calls += 1
    model = judge_model or config.JUDGE_MODEL
    try:
        return llm.chat_json(messages=messages, max_tokens=max_tokens, model=model)
    except Exception as exc:
        logger.debug("Judge model call failed: %s", exc)
        return {}


def context_precision(
    question: str,
    contexts: list[str],
    judge_model: str | None = None,
) -> float | None:
    """Calculate rank-weighted precision for retrieved context passages.

    Formula:
        Average of precision@i across all chunks judged useful.
        If no chunks are useful, returns 0.0.

    Args:
        question: User query string.
        contexts: List of retrieved context text excerpts.
        judge_model: Optional model name to override config.JUDGE_MODEL.

    Returns:
        Score in [0.0, 1.0], or None if unparseable / skipped.
    """
    global _judge_skips
    if not contexts:
        return 0.0
    if not question.strip():
        _judge_skips += 1
        return None

    # Truncate excerpts to keep prompt compact and fast
    passages_text = "\n\n".join(
        f"[{i + 1}] {c[:400]}" for i, c in enumerate(contexts[:5])
    )

    sys_prompt = (
        "You are an evaluation judge. For each numbered passage, judge whether it "
        "contains information directly useful for answering the user's question.\n"
        "Output ONLY a JSON object: {\"useful\": [true, false, ...]} with one boolean per passage."
    )
    user_prompt = f"Question: {question}\n\nPassages:\n{passages_text}"

    data = _call_judge(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=100,
        judge_model=judge_model,
    )

    verdicts = data.get("useful") or data.get("verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        _judge_skips += 1
        return None

    bools = [bool(v) for v in verdicts[: len(contexts)]]
    num_useful = sum(1 for b in bools if b)
    if num_useful == 0:
        return 0.0

    cum_hits = 0
    prec_sum = 0.0
    for i, is_useful in enumerate(bools, start=1):
        if is_useful:
            cum_hits += 1
            prec_sum += cum_hits / i

    return min(1.0, max(0.0, prec_sum / num_useful))


def context_recall(
    question: str,
    reference: str,
    contexts: list[str],
    judge_model: str | None = None,
) -> float | None:
    """Calculate fraction of reference answer statements supported by retrieved context.

    Args:
        question: User query string.
        reference: Ground-truth reference answer.
        contexts: List of retrieved context passages.
        judge_model: Optional model name to override config.JUDGE_MODEL.

    Returns:
        Score in [0.0, 1.0], or None if reference is empty or judge response is invalid.
    """
    global _judge_skips
    if not reference or not reference.strip():
        return None  # Unanswerable / no-retrieval questions have no reference

    statements = split_sentences(reference)
    if not statements:
        return None

    if not contexts:
        return 0.0

    combined_ctx = "\n\n".join(c[:500] for c in contexts[:5])
    statements_text = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(statements))

    sys_prompt = (
        "You are an evaluation judge. For each statement in the reference answer, "
        "determine if it is factual and supported by the retrieved context.\n"
        "Output ONLY a JSON object: {\"supported\": [true, false, ...]} with one boolean per statement."
    )
    user_prompt = f"Context:\n{combined_ctx}\n\nStatements:\n{statements_text}"

    data = _call_judge(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=100,
        judge_model=judge_model,
    )

    supported = data.get("supported") or data.get("verdicts")
    if not isinstance(supported, list) or not supported:
        _judge_skips += 1
        return None

    bools = [bool(s) for s in supported[: len(statements)]]
    return sum(1 for b in bools if b) / len(statements)


def faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    judge_model: str | None = None,
) -> float | None:
    """Calculate fraction of claims in generated answer supported by retrieved context.

    Args:
        question: User query string.
        answer: Generated assistant answer text.
        contexts: List of retrieved context passages.
        judge_model: Optional model name to override config.JUDGE_MODEL.

    Returns:
        Score in [0.0, 1.0], or None if unparseable / answer empty.
    """
    global _judge_skips
    if not answer or not answer.strip():
        return None

    claims = split_sentences(answer)
    if not claims:
        return None

    if not contexts:
        # If no context was provided, any factual claim is ungrounded
        return 0.0

    combined_ctx = "\n\n".join(c[:500] for c in contexts[:5])
    claims_text = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(claims))

    sys_prompt = (
        "You are an evaluation judge. For each claim in the assistant answer, "
        "determine whether it is directly supported by the context.\n"
        "Output ONLY a JSON object: {\"supported\": [true, false, ...]} with one boolean per claim."
    )
    user_prompt = f"Context:\n{combined_ctx}\n\nClaims:\n{claims_text}"

    data = _call_judge(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=100,
        judge_model=judge_model,
    )

    supported = data.get("supported") or data.get("verdicts")
    if not isinstance(supported, list) or not supported:
        _judge_skips += 1
        return None

    bools = [bool(s) for s in supported[: len(claims)]]
    return sum(1 for b in bools if b) / len(claims)


def answer_relevancy(
    question: str,
    answer: str,
    judge_model: str | None = None,
) -> float | None:
    """Calculate cosine similarity between query and questions generated from answer.

    Prompts the judge model to write 3 questions that the answer addresses, embeds
    them alongside the original query, and computes the mean cosine similarity.

    Args:
        question: Original user query string.
        answer: Generated assistant answer.
        judge_model: Optional model name to override config.JUDGE_MODEL.

    Returns:
        Score in [0.0, 1.0], or None if judge output is malformed.
    """
    global _judge_skips
    if not answer or not answer.strip() or not question.strip():
        _judge_skips += 1
        return None

    sys_prompt = (
        "You are an evaluation judge. Based on the given answer, write 3 distinct questions "
        "that this answer addresses directly.\n"
        "Output ONLY a JSON object: {\"questions\": [\"q1\", \"q2\", \"q3\"]}"
    )
    user_prompt = f"Answer:\n{answer[:800]}"

    data = _call_judge(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=150,
        judge_model=judge_model,
    )

    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        _judge_skips += 1
        return None

    clean_qs = [str(q).strip() for q in questions if str(q).strip()]
    if not clean_qs:
        _judge_skips += 1
        return None

    try:
        q_vec = llm.embed([question], kind="query")[0]
        gen_vecs = llm.embed(clean_qs, kind="query")
        # llm.embed produces L2-normalized vectors; dot product gives cosine similarity
        sims = np.dot(gen_vecs, q_vec)
        score = float(np.mean(sims))
        return float(min(1.0, max(0.0, score)))
    except Exception as exc:
        logger.debug("Failed to embed questions for relevancy: %s", exc)
        _judge_skips += 1
        return None
