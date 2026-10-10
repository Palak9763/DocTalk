"""Tests for rag/config.py configuration and presets."""

import importlib
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so imports work when executed directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.config as config


def test_defaults_and_presets() -> None:
    """Verify default settings load correctly and presets are complete."""
    # Project root and paths
    assert config.ROOT == ROOT
    assert config.NOTES_DIR == ROOT / "data" / "notes"
    assert config.LIBRARY_DIR == ROOT / "data" / "library"
    assert config.INDEX_DIR == ROOT / "index"

    # Default model settings
    assert config.OLLAMA_HOST == "http://localhost:11434"
    assert config.CHAT_MODEL == "qwen2.5:3b-instruct"
    assert config.EMBED_MODEL == "nomic-embed-text"
    assert config.RERANK_MODEL == "BAAI/bge-reranker-v2-m3"
    assert config.JUDGE_MODEL == "qwen2.5:3b-instruct"

    # Default speed knobs
    assert config.NUM_CTX == 6144
    assert config.KEEP_ALIVE == "30m"
    assert config.MAX_ANSWER_TOKENS == 600
    assert config.RERANK_MAX_LEN == 384
    assert config.RERANK_TOP_N == 50
    assert config.ANN_MIN_CHUNKS == 20000

    # Embed prefixes
    assert config.EMBED_PREFIX == {
        "document": "search_document: ",
        "query": "search_query: ",
    }

    # Presets definition
    assert set(config.PRESETS.keys()) == {"fast", "balanced", "thorough"}


def test_preset_behaviors() -> None:
    """Verify fast and thorough presets have expected ablation flags."""
    fast = config.PRESETS["fast"]
    assert fast.use_rerank is False
    assert fast.use_expansion is False
    assert fast.use_grading is False

    thorough = config.PRESETS["thorough"]
    assert thorough.grading_mode == "llm"
    assert thorough.use_groundedness_check is True
    assert thorough.n_expansions == 3


def test_env_override() -> None:
    """Verify environment variable overrides are respected upon reload."""
    original_val = os.environ.get("RERANK_TOP_N")
    try:
        os.environ["RERANK_TOP_N"] = "20"
        importlib.reload(config)
        assert config.RERANK_TOP_N == 20
        # PipelineConfig default rerank_top_n should also reflect the reloaded RERANK_TOP_N
        cfg = config.PipelineConfig()
        assert cfg.rerank_top_n == 20
    finally:
        if original_val is not None:
            os.environ["RERANK_TOP_N"] = original_val
        else:
            os.environ.pop("RERANK_TOP_N", None)
        importlib.reload(config)
        assert config.RERANK_TOP_N == 50


def test_preset_labels_differ() -> None:
    """Verify label() generates unique identifiers for all presets."""
    labels = {name: preset.label() for name, preset in config.PRESETS.items()}
    assert len(set(labels.values())) == 3, f"Labels must be distinct: {labels}"
    assert "fast" in labels
    assert "balanced" in labels
    assert "thorough" in labels


def main() -> None:
    test_defaults_and_presets()
    test_preset_behaviors()
    test_env_override()
    test_preset_labels_differ()
    print("All config tests passed successfully!")


if __name__ == "__main__":
    main()
