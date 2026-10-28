"""Configuration and pipeline toggles for Chat With Your Docs."""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env file; real environment variables win (override=False)
load_dotenv(ROOT / ".env", override=False)

def _resolve_dir(env_val: str | None, default: Path) -> Path:
    if not env_val:
        return default
    p = Path(env_val)
    return p if p.is_absolute() else ROOT / p


# Paths
NOTES_DIR: Path = _resolve_dir(os.getenv("NOTES_DIR"), ROOT / "data" / "notes")  # Lecture notes source directory: local raw documents
LIBRARY_DIR: Path = _resolve_dir(os.getenv("LIBRARY_DIR"), ROOT / "data" / "library")  # Uploaded document library: storage vs ephemeral handling
INDEX_DIR: Path = _resolve_dir(os.getenv("INDEX_DIR"), ROOT / "index")  # Embedding index cache: persistence vs recomputation time

# Vector store backend settings
VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "memory").strip().lower()
QDRANT_URL: str = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: Path = _resolve_dir(os.getenv("QDRANT_PATH"), ROOT / "data" / "qdrant")
QDRANT_COLLECTION_PREFIX: str = os.getenv("QDRANT_COLLECTION_PREFIX", "cwyd").strip()

# Model settings
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Ollama endpoint: local daemon vs remote GPU host
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "qwen2.5:3b-instruct")  # Main generator LLM: 3B inference speed vs complex reasoning depth
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")  # Dense embedding model: 8k context & fast inference vs MTEB score
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # Cross-encoder: high multilingual accuracy vs compute latency
JUDGE_MODEL: str = os.getenv("EVAL_JUDGE_MODEL", CHAT_MODEL)  # Evaluation LLM: larger judge yields higher fidelity at slower speed

# Speed knobs
NUM_CTX: int = int(os.getenv("NUM_CTX", "6144"))  # LLM context window tokens: context capacity vs VRAM and latency
KEEP_ALIVE: str = os.getenv("KEEP_ALIVE", "30m")  # Model memory retention: eliminates warm-up reload delay vs freeing memory
MAX_ANSWER_TOKENS: int = int(os.getenv("MAX_ANSWER_TOKENS", "600"))  # Generation length cap: concise output vs detailed responses
RERANK_MAX_LEN: int = int(os.getenv("RERANK_MAX_LEN", "384"))  # Reranker max tokens: cross-attention speed vs long passage fidelity
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "50"))  # Candidates sent to cross-encoder: recall pool depth vs reranking latency
ANN_MIN_CHUNKS: int = int(os.getenv("ANN_MIN_CHUNKS", "20000"))  # Threshold for exact search vs HNSW: exact recall vs build overhead

# Task prefixes for nomic-embed-text
EMBED_PREFIX: dict[str, str] = {
    "document": "search_document: ",  # Prefix for indexing passage embeddings
    "query": "search_query: ",  # Prefix for search query embeddings
}


@dataclass
class PipelineConfig:
    """Configurable toggles for RAG pipeline ablations and runtime tuning."""

    chunk_words: int = 180  # Target words per chunk: context coherence vs retrieval granularity
    chunk_overlap: int = 40  # Word overlap between chunks: continuity across boundaries vs redundancy
    use_rewrite: bool = True  # Query rewrite: resolves conversational references at cost of 1 LLM call
    use_expansion: bool = True  # Query expansion: generates multi-angle queries for recall at cost of 1 LLM call
    n_expansions: int = 2  # Number of expanded query variants: broader coverage vs retrieval overhead
    use_hyde: bool = False  # Hypothetical document embeddings: zero-shot alignment at cost of 1 LLM call
    retrieval: str = "hybrid"  # Retrieval mode ("dense" | "sparse" | "hybrid"): semantic vs exact term matching
    dense_k: int = 30  # Dense candidates fetched: semantic candidate recall vs embedding search latency
    sparse_k: int = 30  # Sparse BM25 candidates fetched: keyword candidate recall vs inverted index lookup
    rrf_k: int = 60  # RRF constant: balances top-rank impact vs tail-rank distribution
    max_candidates: int = 50  # Deduplicated candidate pool size after fusion: candidate diversity vs reranker load
    use_stemming: bool = True  # Snowball stemming: inflectional matching vs stem collision risk
    dedupe_content: bool = True  # Content deduplication: eliminates duplicate chunks vs preserving duplicates
    use_rerank: bool = True  # Cross-encoder reranker: precision ranking vs cross-attention compute
    rerank_top_n: int = RERANK_TOP_N  # Chunks passed to reranker: candidate coverage vs reranker runtime
    final_k: int = 5  # Final chunks passed to generator: prompt token usage vs grounding context depth
    use_adaptive: bool = True  # Self-RAG routing: skips retrieval for greetings/direct queries saving latency
    use_grading: bool = True  # Chunk relevance grading: filters noise chunks prior to generation
    grading_mode: str = "rerank"  # Grading method ("rerank" | "llm"): fast score thresholding vs LLM reflection
    rerank_min_score: float = 0.05  # Minimum rerank threshold: noise rejection vs retaining borderline chunks
    use_groundedness_check: bool = False  # Post-generation citation check: hallucination detection at cost of 1 LLM call

    def label(self) -> str:
        """Return a compact human-readable descriptor of the active pipeline toggles."""
        parts: list[str] = [self.retrieval]
        if self.use_rewrite:
            parts.append("rewrite")
        if self.use_expansion:
            parts.append("expand")
        if self.use_hyde:
            parts.append("hyde")
        if self.use_rerank:
            parts.append("rerank")
        if self.use_adaptive:
            parts.append("selfrag")
        if self.use_grading:
            parts.append(f"grade_{self.grading_mode}" if self.grading_mode else "grade")
        if self.use_groundedness_check:
            parts.append("grounded")
        return "+".join(parts)


PRESETS: dict[str, PipelineConfig] = {
    # Fast preset (~2 LLM calls: adaptive routing + query rewrite + answer generation)
    "fast": PipelineConfig(
        use_expansion=False,
        use_rerank=False,
        use_grading=False,
    ),
    # Balanced preset (~3 LLM calls: adaptive routing + rewrite + expansion + answer generation)
    "balanced": PipelineConfig(),
    # Thorough preset (~9 LLM calls: adaptive routing + rewrite + 3 expansions + LLM grading x5 + answer + groundedness)
    "thorough": PipelineConfig(
        n_expansions=3,
        grading_mode="llm",
        use_groundedness_check=True,
    ),
}
