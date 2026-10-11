"""Query processing module: conversational query rewriting, multi-query expansion, and HyDE passages."""

import re
from typing import Any

from rag import llm


def _format_history_turn(turn: Any) -> str:
    """Format a single conversation history turn into a 'role: content' string."""
    if isinstance(turn, (tuple, list)) and len(turn) >= 2:
        return f"{turn[0]}: {turn[1]}"
    if isinstance(turn, dict):
        role = turn.get("role", "user")
        content = turn.get("content", "")
        return f"{role}: {content}"
    return str(turn)


def rewrite_query(
    question: str,
    history: list[Any] | None = None,
    max_tokens: int = 80,
) -> str:
    """Rewrite a follow-up question into a standalone question using conversation history.

    Args:
        question: Current user follow-up question.
        history: Conversation history (list of (role, content) tuples or dicts).
        max_tokens: Hard token generation budget.

    Returns:
        Single-line standalone question. Falls back to original question if history is empty,
        model returns empty output, or generated output exceeds ~3x question length.
    """
    if not history or not question or not question.strip():
        return question

    recent_history = history[-6:]
    history_lines = [_format_history_turn(turn) for turn in recent_history]
    history_text = "\n".join(history_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query reformulation assistant. Rewrite the user's follow-up question "
                "into an independent, standalone question using conversation history to resolve pronouns and references.\n"
                "Format: Output only the single rewritten question without quotes, explanations, or prefixes.\n"
                'Example: History: "user: What is backpropagation? | assistant: It computes gradients using the chain rule." '
                'Follow-up: "why does it need the chain rule?" -> Standalone: Why does backpropagation need the chain rule?'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation history:\n{history_text}\n\n"
                f"Follow-up question: {question}\n"
                "Standalone question:"
            ),
        },
    ]

    try:
        raw_response = llm.chat(messages=messages, temperature=0.0, max_tokens=max_tokens)
    except Exception:
        return question

    if not raw_response or not raw_response.strip():
        return question

    # Process first non-empty line
    lines = [line.strip() for line in raw_response.strip().splitlines() if line.strip()]
    if not lines:
        return question

    cleaned = lines[0]
    # Strip quotes
    cleaned = cleaned.strip("\"'“”`")
    # Strip common prefixes like 'Rewritten question:', 'Standalone question:', etc.
    cleaned = re.sub(
        r"^(?:rewritten question|standalone question|rewritten|question)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = cleaned.strip("\"'“”`")

    if not cleaned:
        return question

    # Length guard: if more than ~3x longer than the question, fall back
    if len(cleaned) > 3 * max(len(question), 1):
        return question

    return cleaned


def expand_query(
    query: str,
    n: int = 2,
    max_tokens: int = 120,
) -> list[str]:
    """Generate n alternative phrasings or related search queries via chat_json.

    Args:
        query: Original search query.
        n: Number of alternative queries to return.
        max_tokens: Hard token generation budget.

    Returns:
        List of at most n distinct non-empty string queries, excluding the original query.
        Returns [] on failure.
    """
    if n <= 0 or not query or not query.strip():
        return []

    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query expansion assistant. Generate alternative phrasings "
                "or closely related search queries for document retrieval.\n"
                'Format: Return only valid JSON: {"queries": ["query 1", "query 2"]}\n'
                'Example: For "backpropagation formula", return {"queries": ["backpropagation equation", "chain rule gradient calculation"]}'
            ),
        },
        {
            "role": "user",
            "content": f'Generate {n} alternative search queries for: "{query}"',
        },
    ]

    try:
        data = llm.chat_json(messages=messages, max_tokens=max_tokens)
    except Exception:
        return []

    items: list[Any] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        val = data.get("queries")
        if isinstance(val, list):
            items = val
        elif not val and len(data) == 1 and isinstance(list(data.values())[0], list):
            items = list(data.values())[0]

    normalized_orig = query.strip().lower()
    results: list[str] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        cleaned_lower = cleaned.lower()
        if cleaned_lower == normalized_orig:
            continue
        if cleaned_lower in seen:
            continue
        seen.add(cleaned_lower)
        results.append(cleaned)
        if len(results) == n:
            break

    return results


def hyde_passage(
    query: str,
    max_tokens: int = 160,
) -> str:
    """Generate a short hypothetical lecture note answer passage (2-3 sentences) for embedding.

    Args:
        query: Search query to answer hypothetically.
        max_tokens: Hard token generation budget.

    Returns:
        Hypothetical passage text, or empty string on failure.
    """
    if not query or not query.strip():
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic lecture note assistant. Write a short, hypothetical answer passage "
                "(2-3 sentences) as it would appear in lecture notes to answer the query.\n"
                "Format: Return only the 2-3 sentence lecture note passage text without commentary or preamble.\n"
                'Example: Query: "What is backpropagation?" -> Passage: Backpropagation computes loss gradients '
                "layer by layer using the chain rule. It propagates error terms backward from the output layer to update network weights."
            ),
        },
        {
            "role": "user",
            "content": f"Query: {query}\nLecture passage:",
        },
    ]

    try:
        raw_response = llm.chat(messages=messages, temperature=0.3, max_tokens=max_tokens)
    except Exception:
        return ""

    if not raw_response or not raw_response.strip():
        return ""

    cleaned = raw_response.strip()
    cleaned = re.sub(r"^(?:passage|lecture passage)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.strip("\"'“”`")
