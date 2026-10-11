"""Self-RAG module: adaptive retrieval decision, chunk grading, and groundedness checking."""

import logging
from typing import Any

from rag import llm
from rag.ingest import Chunk

logger = logging.getLogger(__name__)

# Tokens caps – small enough for a 3B model to handle quickly
_DECIDE_TOKENS = 60
_GRADE_TOKENS = 40
_GROUND_TOKENS = 200


def needs_retrieval(
    question: str,
    history: list[Any] | None = None,
) -> tuple[bool, str]:
    """Decide whether retrieval is needed to answer the question.

    Retrieval is the DEFAULT. Returns False ONLY for clear small-talk / greetings
    with no reference to documents or any topic-looking noun.

    Args:
        question: User's current question.
        history: Optional conversation history (unused in decision, for context only).

    Returns:
        (retrieve: bool, reason: str). Unparseable output, missing key, or any
        exception defaults to (True, "defaulted to retrieval").
    """
    if not question or not question.strip():
        return True, "empty question – defaulted to retrieval"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a routing assistant. Decide if the user's message requires searching documents.\n"
                'Return ONLY valid JSON: {"retrieve": true, "reason": "one sentence"}\n'
                "Return retrieve=false ONLY for obvious small-talk or greetings with NO topic noun "
                "(e.g. 'hi', 'thanks', 'who are you'). For ANY question about a concept, topic, or fact return true.\n"
                'Example: {"retrieve": false, "reason": "Pure greeting, no document topic."}'
            ),
        },
        {
            "role": "user",
            "content": f"User message: {question}",
        },
    ]

    try:
        data = llm.chat_json(messages=messages, max_tokens=_DECIDE_TOKENS)
    except Exception as exc:
        logger.warning("needs_retrieval LLM call failed: %s", exc)
        return True, "defaulted to retrieval"

    if not isinstance(data, dict) or "retrieve" not in data:
        return True, "defaulted to retrieval"

    try:
        retrieve = bool(data["retrieve"])
        reason = str(data.get("reason", ""))
        return retrieve, reason
    except Exception:
        return True, "defaulted to retrieval"


def grade_chunks(
    query: str,
    chunks: list[Chunk],
) -> list[bool]:
    """Grade each chunk for relevance to the query using one LLM call per chunk.

    A failed or unparseable verdict always keeps the chunk (returns True).
    Never drops evidence because the model misbehaved.

    Args:
        query: The search query the chunks were retrieved for.
        chunks: List of Chunk objects to evaluate.

    Returns:
        List of bools, same length as chunks. True = relevant (keep), False = irrelevant (drop).
    """
    if not chunks:
        return []

    results: list[bool] = []
    for chunk in chunks:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance grader. Decide if the excerpt answers or supports the query.\n"
                    'Return ONLY valid JSON: {"relevant": true} or {"relevant": false}\n'
                    'Example: {"relevant": true}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Excerpt:\n{chunk.body[:600]}"
                ),
            },
        ]

        try:
            data = llm.chat_json(messages=messages, max_tokens=_GRADE_TOKENS)
        except Exception as exc:
            logger.warning("grade_chunks LLM call failed for chunk %s: %s", chunk.id, exc)
            results.append(True)  # keep on failure
            continue

        if not isinstance(data, dict) or "relevant" not in data:
            results.append(True)  # keep on unparseable output
            continue

        try:
            results.append(bool(data["relevant"]))
        except Exception:
            results.append(True)

    return results


def check_groundedness(
    answer: str,
    chunks: list[Chunk],
) -> dict[str, Any]:
    """Check whether every claim in the answer is supported by the provided excerpts.

    Args:
        answer: Generated answer text.
        chunks: Chunks that were used as context for generation.

    Returns:
        {"grounded": bool, "unsupported": list[str]}
        Unparseable output returns {"grounded": True, "unsupported": []} to avoid
        triggering needless regeneration.
    """
    _default = {"grounded": True, "unsupported": []}
    if not answer or not chunks:
        return _default

    excerpts = "\n\n".join(
        f"[{c.id}] {c.body[:400]}" for c in chunks
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a groundedness checker. Determine if every claim in the answer is supported by the excerpts.\n"
                'Return ONLY valid JSON: {"grounded": true, "unsupported": []} or '
                '{"grounded": false, "unsupported": ["claim 1 not found", "claim 2 not found"]}\n'
                'Example: {"grounded": true, "unsupported": []}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Excerpts:\n{excerpts}\n\n"
                f"Answer:\n{answer}\n\n"
                "Is every claim in the answer supported by the excerpts above?"
            ),
        },
    ]

    try:
        data = llm.chat_json(messages=messages, max_tokens=_GROUND_TOKENS)
    except Exception as exc:
        logger.warning("check_groundedness LLM call failed: %s", exc)
        return _default

    if not isinstance(data, dict) or "grounded" not in data:
        return _default

    try:
        grounded = bool(data["grounded"])
        unsupported = data.get("unsupported", [])
        if not isinstance(unsupported, list):
            unsupported = []
        unsupported = [str(s) for s in unsupported if s]
        return {"grounded": grounded, "unsupported": unsupported}
    except Exception:
        return _default
