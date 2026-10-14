"""Local Ollama client wrapper for chat generation, streaming, and cached embeddings."""

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Generator, Literal

import numpy as np
import ollama

from rag import config

logger = logging.getLogger(__name__)

# Lazily initialized module-level client
_client: ollama.Client | None = None

# Two-tier embedding caches
_doc_cache: dict[str, list[float]] = {}
_query_cache: dict[str, list[float]] = {}
_doc_cache_loaded: bool = False


def get_client() -> ollama.Client:
    """Return the global Ollama client, instantiating it if necessary."""
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client


def _extract_content(msg: Any) -> str:
    """Extract text content from an Ollama response or dict object."""
    if isinstance(msg, dict):
        return msg.get("message", {}).get("content", "")
    if hasattr(msg, "message"):
        m = msg.message
        if isinstance(m, dict):
            return m.get("content", "")
        if hasattr(m, "content"):
            return str(m.content)
    return ""


def chat(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> str:
    """Execute a chat completion request via Ollama."""
    client = get_client()
    target_model = model or config.CHAT_MODEL
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_ctx": config.NUM_CTX,
        "num_predict": max_tokens,
    }
    response = client.chat(
        model=target_model,
        messages=messages,
        stream=False,
        options=options,
        keep_alive=config.KEEP_ALIVE,
    )
    return _extract_content(response)


def chat_stream(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> Generator[str, None, None]:
    """Stream chat completion text chunks via Ollama."""
    client = get_client()
    target_model = model or config.CHAT_MODEL
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_ctx": config.NUM_CTX,
        "num_predict": max_tokens,
    }
    response = client.chat(
        model=target_model,
        messages=messages,
        stream=True,
        options=options,
        keep_alive=config.KEEP_ALIVE,
    )
    for chunk in response:
        content = _extract_content(chunk)
        if content:
            yield content


def parse_json(text: str) -> dict:
    """Parse a JSON object from text, tolerating markdown fences and chatter.

    Returns:
        Parsed dict, or {} for non-objects, arrays, or malformed inputs.
    """
    if not text or not isinstance(text, str):
        return {}

    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        prefix = cleaned[:start].strip()
        suffix = cleaned[end + 1 :].strip()
        if prefix.startswith("[") and suffix.endswith("]"):
            return {}
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {}


def chat_json(
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    model: str | None = None,
) -> dict:
    """Call chat at temperature 0 and parse the resulting JSON object."""
    response = chat(
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        model=model,
    )
    return parse_json(response)


def _load_doc_cache_if_needed() -> None:
    """Load persistent document embedding cache from disk if not already loaded."""
    global _doc_cache, _doc_cache_loaded
    if not _doc_cache_loaded:
        cache_file = Path(config.INDEX_DIR) / "embed_cache.json"
        if cache_file.is_file():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    _doc_cache = data
                else:
                    logger.warning("Corrupt embed cache (not a dict), starting empty")
                    _doc_cache = {}
            except Exception as exc:
                logger.warning("Failed to load embed cache (%s), starting empty", exc)
                _doc_cache = {}
        else:
            _doc_cache = {}
        _doc_cache_loaded = True


def _save_doc_cache() -> None:
    """Persist document embedding cache to INDEX_DIR/embed_cache.json."""
    index_dir = Path(config.INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)
    cache_file = index_dir / "embed_cache.json"
    cache_file.write_text(json.dumps(_doc_cache), encoding="utf-8")


def reset_embedding_cache() -> None:
    """Clear both in-memory embedding caches and reset disk-loaded state."""
    global _doc_cache, _query_cache, _doc_cache_loaded
    _doc_cache.clear()
    _query_cache.clear()
    _doc_cache_loaded = False


def embed(
    texts: list[str], kind: Literal["document", "query"] = "document"
) -> np.ndarray:
    """Generate L2-normalized float32 embeddings for a list of texts.

    Uses Ollama embed endpoint with task prefixes, batching, and a two-tier cache
    (documents persisted to INDEX_DIR/embed_cache.json, queries in memory only).

    Args:
        texts: List of text strings to embed.
        kind: "document" for passages or "query" for search queries.

    Returns:
        np.ndarray of shape (n, dim) with L2-normalized float32 rows.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = config.EMBED_MODEL
    prefix = config.EMBED_PREFIX.get(kind, "")
    _load_doc_cache_if_needed()

    prefixed_texts = [f"{prefix}{t}" for t in texts]
    keys = [
        hashlib.md5(f"{pt}{model}".encode("utf-8")).hexdigest()
        for pt in prefixed_texts
    ]

    target_cache = _doc_cache if kind == "document" else _query_cache
    missing_indices = [i for i, k in enumerate(keys) if k not in target_cache]

    if missing_indices:
        client = get_client()
        batch_size = 32
        for start_idx in range(0, len(missing_indices), batch_size):
            batch_slice = missing_indices[start_idx : start_idx + batch_size]
            batch_inputs = [prefixed_texts[i] for i in batch_slice]
            batch_keys = [keys[i] for i in batch_slice]

            response = client.embed(
                model=model,
                input=batch_inputs,
                keep_alive=config.KEEP_ALIVE,
            )
            embs = (
                response.embeddings
                if hasattr(response, "embeddings")
                else response["embeddings"]
            )
            for k, emb in zip(batch_keys, embs):
                target_cache[k] = list(emb)

        if kind == "document":
            _save_doc_cache()

    vectors = [target_cache[k] for k in keys]
    matrix = np.array(vectors, dtype=np.float32)
    if matrix.ndim == 2 and matrix.shape[0] > 0 and matrix.shape[1] > 0:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

    return matrix.astype(np.float32)


def warm_up() -> None:
    """Preload chat and embedding models in memory; swallow and log any errors."""
    try:
        chat([{"role": "user", "content": "hi"}], max_tokens=1)
    except Exception as exc:
        logger.warning("Failed to warm up chat model: %s", exc)
    try:
        embed(["warmup"], kind="document")
    except Exception as exc:
        logger.warning("Failed to warm up embedding model: %s", exc)
