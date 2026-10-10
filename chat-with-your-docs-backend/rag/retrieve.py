"""Core retrieval module: lexical analysis, BM25 inverted index, dense vector index, and hybrid RRF fusion."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import re
import threading
from typing import Any, Literal

import numpy as np

try:
    import snowballstemmer

    _stemmer: Any = snowballstemmer.stemmer("english")
except ImportError:
    _stemmer = None

try:
    import hnswlib
except ImportError:
    hnswlib = None

from rag.config import ANN_MIN_CHUNKS
from rag.ingest import Chunk
from rag.llm import embed

# English stopword set (~60 words: articles, prepositions, auxiliaries, wh-words)
STOPWORDS: frozenset[str] = frozenset({
    # Articles
    "a", "an", "the",
    # Prepositions
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beside", "between", "beyond", "by",
    "down", "during", "for", "from", "in", "into", "near", "of", "off", "on",
    "onto", "out", "over", "through", "to", "toward", "under", "until", "up",
    "upon", "with", "within", "without",
    # Auxiliaries & be-verbs
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",
    # Conjunctions & connectives
    "and", "but", "or", "nor", "so", "yet", "if", "than", "then", "else",
    "when", "while", "as", "because", "since",
    # Pronouns & demonstratives
    "it", "its", "this", "that", "these", "those",
    # WH-words
    "who", "whom", "whose", "which", "what", "where", "why", "how",
})

_stem_lock = threading.Lock()


@lru_cache(maxsize=10000)
def _cached_stem(token: str) -> str:
    """Stem a single word using Snowball Porter2 stemmer with thread synchronization."""
    if _stemmer is None:
        return token
    with _stem_lock:
        return _stemmer.stemWord(token)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words minus English stopwords.

    Args:
        text: Raw input text.

    Returns:
        List of non-stopword tokens.
    """
    if not text:
        return []
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def analyze(text: str, stem: bool = True) -> list[str]:
    """Analyze text with tokenization, stopword removal, and optional Snowball stemming.

    Args:
        text: Raw text to analyze.
        stem: Whether to apply Snowball stemming (if snowballstemmer is installed).

    Returns:
        List of analyzed tokens or stems.
    """
    tokens = tokenize(text)
    if not stem or _stemmer is None:
        return tokens
    return [_cached_stem(t) for t in tokens]


class InvertedIndex:
    """Inverted index implementing Okapi BM25 with Lucene-style non-negative IDF."""

    def __init__(
        self,
        docs: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = (
            sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0.0
        )

        # Term -> list of (doc_id, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            term_counts = Counter(doc)
            for term, count in term_counts.items():
                self.postings[term].append((doc_id, count))

        # Lucene-style non-negative IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        self.idf: dict[str, float] = {}
        for term, plist in self.postings.items():
            df = len(plist)
            self.idf[term] = math.log(
                1.0 + (self.n_docs - df + 0.5) / (df + 0.5)
            )

    def search(self, tokens: list[str], k: int) -> list[tuple[int, float]]:
        """Score and rank documents matching query tokens using Okapi BM25.

        Args:
            tokens: Analyzed query terms.
            k: Maximum number of top documents to return.

        Returns:
            List of (doc_id, score) tuples sorted best first. Ties are broken
            by lower doc index. Only documents matching at least one query term
            are scored; returns [] if no terms match.
        """
        if k <= 0 or not tokens or self.n_docs == 0:
            return []

        doc_scores: dict[int, float] = defaultdict(float)
        query_terms = set(tokens)

        for term in query_terms:
            if term not in self.postings:
                continue
            term_idf = self.idf[term]
            for doc_id, tf in self.postings[term]:
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (
                    1.0
                    - self.b
                    + self.b * (dl / self.avgdl if self.avgdl > 0 else 1.0)
                )
                doc_scores[doc_id] += term_idf * (tf * (self.k1 + 1.0)) / denom

        if not doc_scores:
            return []

        ranked = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]


class DenseIndex:
    """Dense vector index with exact cosine matrix search and HNSW ANN scaling."""

    def __init__(
        self, vecs: np.ndarray, ann_min: int = ANN_MIN_CHUNKS
    ) -> None:
        self.vecs = np.asarray(vecs, dtype=np.float32)
        self.n = len(self.vecs) if self.vecs.ndim == 2 else 0
        self.ann_min = ann_min
        self._hnsw_index: Any = None
        self._lock = threading.Lock()

        if hnswlib is not None and self.n >= ann_min and self.n > 0:
            dim = self.vecs.shape[1]
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=self.n, ef_construction=200, M=16)
            index.add_items(self.vecs, np.arange(self.n))
            self._hnsw_index = index
            self.kind = "HNSW (approximate)"
        else:
            self.kind = "exact cosine"

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Search top-k most similar vectors by cosine similarity.

        Args:
            qvec: Query vector.
            k: Maximum candidates to retrieve (clamped to corpus size n).

        Returns:
            List of (doc_index, similarity) sorted best first.
        """
        if self.n == 0 or k <= 0:
            return []

        k = min(k, self.n)
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        if self._hnsw_index is not None:
            with self._lock:
                self._hnsw_index.set_ef(max(64, 2 * k))
                labels, distances = self._hnsw_index.knn_query(q, k=k)
            results = [
                (int(lbl), float(1.0 - dist))
                for lbl, dist in zip(labels[0], distances[0])
            ]
            results.sort(key=lambda item: (-item[1], item[0]))
            return results
        else:
            sims = self.vecs.dot(q)
            ranked_indices = sorted(
                range(self.n), key=lambda idx: (-float(sims[idx]), idx)
            )
            return [(int(idx), float(sims[idx])) for idx in ranked_indices[:k]]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Score formula: score(d) = sum_{lists} 1 / (k + rank), where rank is 1-based.

    Args:
        rankings: List of ranked doc index lists.
        k: RRF constant factor.

    Returns:
        Consolidated list of (doc_id, score) sorted by score desc, then doc_id asc.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank_0, doc_id in enumerate(ranking):
            rank = rank_0 + 1
            scores[doc_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass
class SearchResult:
    """Container for hybrid retrieval candidates, path ranks, and deduplication statistics."""

    candidates: list[int]
    dense_rank: dict[int, int]
    sparse_rank: dict[int, int]
    fused_total: int
    duplicates_removed: int


class Retriever:
    """Hybrid dense and sparse retriever with Reciprocal Rank Fusion and fingerprint deduplication."""

    def __init__(
        self,
        chunks: list[Chunk],
        use_dense: bool = True,
        stem: bool = True,
        ann_min: int = ANN_MIN_CHUNKS,
    ) -> None:
        self.chunks = chunks
        self.stem = stem
        self.use_dense = use_dense

        # 1. Build sparse inverted index from analyzed chunk texts
        analyzed_docs = [analyze(c.text, stem=stem) for c in chunks]
        self.inverted_index = InvertedIndex(analyzed_docs)

        # 2. Build dense index if enabled
        if use_dense and len(chunks) > 0:
            texts = [c.text for c in chunks]
            vecs = embed(texts, kind="document")
            self.dense_index: DenseIndex | None = DenseIndex(
                vecs, ann_min=ann_min
            )
        elif use_dense:
            self.dense_index = DenseIndex(
                np.empty((0, 0), dtype=np.float32), ann_min=ann_min
            )
        else:
            self.dense_index = None

        # 3. MD5 fingerprint per chunk based on whitespace-normalized, lowercased body
        self.fingerprints = [
            hashlib.md5(" ".join(c.body.lower().split()).encode("utf-8")).hexdigest()
            for c in chunks
        ]

    def sparse(self, query: str, k: int = 30) -> list[int]:
        """Perform sparse BM25 search for a query string."""
        tokens = analyze(query, stem=self.stem)
        results = self.inverted_index.search(tokens, k)
        return [doc_id for doc_id, _ in results]

    def dense(self, qvec: np.ndarray, k: int = 30) -> list[int]:
        """Perform dense cosine search for a query vector."""
        if self.dense_index is None:
            raise ValueError("Dense index is not available")
        results = self.dense_index.search(qvec, k)
        return [idx for idx, _ in results]

    def search(
        self,
        queries: list[str] | str,
        mode: Literal["sparse", "dense", "hybrid"] = "hybrid",
        dense_k: int = 30,
        sparse_k: int = 30,
        rrf_k: int = 60,
        max_candidates: int = 50,
        hyde_texts: list[str] | None = None,
        dedupe: bool = True,
    ) -> SearchResult:
        """Search corpus across query variants using specified retrieval mode.

        Args:
            queries: Single query string or list of query variants.
            mode: "sparse", "dense", or "hybrid".
            dense_k: Candidates fetched per dense query variant.
            sparse_k: Candidates fetched per sparse query variant.
            rrf_k: Reciprocal Rank Fusion constant.
            max_candidates: Maximum deduplicated candidate pool size.
            hyde_texts: Optional hypothetical passage variants.
            dedupe: Whether to deduplicate chunks with identical body text.

        Returns:
            SearchResult containing final candidate indices and provenance stats.

        Raises:
            ValueError: If mode is unsupported or mode is 'dense' without a dense index.
        """
        if not self.chunks:
            return SearchResult(
                candidates=[],
                dense_rank={},
                sparse_rank={},
                fused_total=0,
                duplicates_removed=0,
            )

        if mode not in ("sparse", "dense", "hybrid"):
            raise ValueError(f"Unsupported retrieval mode: '{mode}'")

        if mode == "dense" and self.dense_index is None:
            raise ValueError("Mode is 'dense' but Retriever has no dense index")

        query_list = [queries] if isinstance(queries, str) else list(queries)

        all_rankings: list[list[int]] = []
        dense_rank: dict[int, int] = {}
        sparse_rank: dict[int, int] = {}

        # 1. Sparse retrieval path
        if mode in ("sparse", "hybrid"):
            for q in query_list:
                sparse_hits = self.sparse(q, k=sparse_k)
                all_rankings.append(sparse_hits)
                for rank_0, doc_id in enumerate(sparse_hits):
                    rank = rank_0 + 1
                    if doc_id not in sparse_rank or rank < sparse_rank[doc_id]:
                        sparse_rank[doc_id] = rank

        # 2. Dense retrieval path
        if mode in ("dense", "hybrid") and self.dense_index is not None:
            for q in query_list:
                q_vec = embed([q], kind="query")[0]
                dense_hits = self.dense(q_vec, k=dense_k)
                all_rankings.append(dense_hits)
                for rank_0, doc_id in enumerate(dense_hits):
                    rank = rank_0 + 1
                    if doc_id not in dense_rank or rank < dense_rank[doc_id]:
                        dense_rank[doc_id] = rank

            if hyde_texts:
                for ht in hyde_texts:
                    ht_vec = embed([ht], kind="document")[0]
                    dense_hits = self.dense(ht_vec, k=dense_k)
                    all_rankings.append(dense_hits)
                    for rank_0, doc_id in enumerate(dense_hits):
                        rank = rank_0 + 1
                        if doc_id not in dense_rank or rank < dense_rank[doc_id]:
                            dense_rank[doc_id] = rank

        # 3. Reciprocal Rank Fusion
        fused = rrf(all_rankings, k=rrf_k)
        fused_total = len(fused)

        # 4. Fingerprint deduplication and candidate capping
        if dedupe:
            seen_fps: set[str] = set()
            deduped_candidates: list[int] = []
            for doc_id, _ in fused:
                fp = self.fingerprints[doc_id]
                if fp not in seen_fps:
                    seen_fps.add(fp)
                    deduped_candidates.append(doc_id)
            duplicates_removed = fused_total - len(deduped_candidates)
            candidates = deduped_candidates[:max_candidates]
        else:
            duplicates_removed = 0
            candidates = [doc_id for doc_id, _ in fused][:max_candidates]

        return SearchResult(
            candidates=candidates,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            fused_total=fused_total,
            duplicates_removed=duplicates_removed,
        )
