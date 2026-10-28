"""Unit tests for rag/retrieve.py: InvertedIndex, DenseIndex, RRF, and Retriever."""
import math
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import NOTES_DIR
from rag.ingest import Chunk, load_chunks
from rag.retrieve import (
    DenseIndex,
    InvertedIndex,
    Retriever,
    SearchResult,
    analyze,
    rrf,
    tokenize,
)
from tests.fakes import install
def test_inverted_index_bm25_math() -> None:
    """Verify BM25 math matches manual computation, IDF is non-negative, and filtering works."""
    # 3 docs:
    # doc 0: ["deep", "learning"] (dl = 2)
    # doc 1: ["deep", "neural", "network"] (dl = 3)
    # doc 2: ["machine", "learning"] (dl = 2)
    docs = [
        ["deep", "learning"],
        ["deep", "neural", "network"],
        ["machine", "learning"],
    ]
    idx = InvertedIndex(docs, k1=1.5, b=0.75)

    # N = 3, avgdl = 7/3
    # term "deep" appears in doc 0 and doc 1 -> df = 2
    # idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1 + 1.5 / 2.5) = ln(1.6)
    idf_deep = math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5))
    avgdl = 7.0 / 3.0

    # doc 0: tf=1, dl=2
    # B = 1 - 0.75 + 0.75 * (2 / (7/3)) = 0.25 + 1.5 / (7/3) = 0.25 + 4.5/7
    b0 = 0.25 + 0.75 * (2.0 / avgdl)
    denom0 = 1.0 + 1.5 * b0
    expected_s0 = idf_deep * (1.0 * (1.5 + 1.0)) / denom0

    # doc 1: tf=1, dl=3
    b1 = 0.25 + 0.75 * (3.0 / avgdl)
    denom1 = 1.0 + 1.5 * b1
    expected_s1 = idf_deep * (1.0 * (1.5 + 1.0)) / denom1

    results = idx.search(["deep"], k=5)
    assert len(results) == 2, f"Expected 2 matches, got: {results}"
    assert results[0][0] == 0
    assert results[1][0] == 1
    assert math.isclose(results[0][1], expected_s0, rel_tol=1e-5)
    assert math.isclose(results[1][1], expected_s1, rel_tol=1e-5)

    # 1. Only docs containing query term are scored: doc 2 must not be scored
    doc_ids_scored = [doc_id for doc_id, _ in results]
    assert 2 not in doc_ids_scored

    # 2. Unknown terms return empty list
    assert idx.search(["unknownterm"], k=5) == []

    # 3. IDF is never negative even when term is in all documents
    all_docs_term_idx = InvertedIndex([["common"], ["common"], ["common"]])
    assert all_docs_term_idx.idf["common"] > 0.0, "IDF must remain positive"
    res_common = all_docs_term_idx.search(["common"], k=3)
    assert len(res_common) == 3
    assert all(score > 0.0 for _, score in res_common)

def test_analyze() -> None:
    """Verify lowercasing, stopword dropping, and Snowball morphological stemming."""
    # Lowercasing
    assert "gradient" in analyze("GRADIENT")
    # Stopwords dropping
    assert analyze("the and in on of for") == []
    # Stemming "training models" vs "trains model"
    tokens1 = analyze("training models", stem=True)
    tokens2 = analyze("trains model", stem=True)
    assert tokens1 == ["train", "model"]
    assert tokens2 == ["train", "model"]
    assert tokens1 == tokens2

def test_rrf() -> None:
    """Verify reciprocal rank fusion scoring rules and ties."""
    # Single doc at rank 1 with k=60 -> exact score 1 / 61
    single = rrf([[10]], k=60)
    assert len(single) == 1
    assert single[0][0] == 10
    assert math.isclose(single[0][1], 1.0 / 61.0, rel_tol=1e-9)

    # A doc first in both lists wins
    # doc 0 is #1 in list 1 (1/61) and #1 in list 2 (1/61) -> 2/61
    # doc 1 is #2 in list 1 (1/62)
    # doc 2 is #2 in list 2 (1/62)
    fused = rrf([[0, 1], [0, 2]], k=60)
    assert fused[0][0] == 0
    assert math.isclose(fused[0][1], 2.0 / 61.0, rel_tol=1e-9)

    # A doc in both lists beats a doc in one list at equal rank
    # doc 3 in both lists at rank 2: 1/62 + 1/62 = 2/62 ≈ 0.03225
    # doc 4 only in list 1 at rank 1: 1/61 ≈ 0.01639
    fused2 = rrf([[4, 3], [5, 3]], k=60)
    scores_dict = dict(fused2)
    assert scores_dict[3] > scores_dict[4]

def test_search_shape() -> None:
    """Verify candidate count limits, deduplication, path ranks, and mode error handling."""
    with tempfile.TemporaryDirectory() as td:
        install(td)
        # 6 chunks: c0 and c2 share identical body content
        chunks = [
            Chunk(
                id="c0",
                source="doc1.md",
                section="SecA",
                text="doc1.md › SecA\nMachine learning gradient descent optimization algorithm",
                page=None,
                prefix_len=15,
            ),
            Chunk(
                id="c1",
                source="doc1.md",
                section="SecB",
                text="doc1.md › SecB\nDeep convolutional neural network architecture layers",
                page=None,
                prefix_len=15,
            ),
            Chunk(
                id="c2",
                source="doc2.md",
                section="SecA",
                text="doc2.md › SecA\nMachine learning gradient descent optimization algorithm",
                page=None,
                prefix_len=15,
            ),
            Chunk(
                id="c3",
                source="doc2.md",
                section="SecB",
                text="doc2.md › SecB\nRecurrent long short term memory sequence modeling",
                page=None,
                prefix_len=15,
            ),
            Chunk(
                id="c4",
                source="doc3.md",
                section="SecA",
                text="doc3.md › SecA\nTransformer multi head self attention mechanism",
                page=None,
                prefix_len=15,
            ),
            Chunk(
                id="c5",
                source="doc3.md",
                section="SecB",
                text="doc3.md › SecB\nReinforcement learning policy gradient actor critic",
                page=None,
                prefix_len=15,
            ),
        ]

        retriever = Retriever(chunks, use_dense=True)
        query = "machine learning gradient descent"

        # Search with dense_k=3, sparse_k=3, max_candidates=4, dedupe=True
        res = retriever.search(
            query,
            mode="hybrid",
            dense_k=3,
            sparse_k=3,
            max_candidates=4,
            dedupe=True,
        )

        # 1. candidates <= 4
        assert len(res.candidates) <= 4

        # 2. No duplicates in candidates
        assert len(set(res.candidates)) == len(res.candidates)

        # 3. Identical-text chunks appear once
        has_c0 = 0 in res.candidates
        has_c2 = 2 in res.candidates
        assert (has_c0 or has_c2) and not (
            has_c0 and has_c2
        ), f"Expected exactly one copy of identical chunks, got: {res.candidates}"

        # 4. duplicates_removed >= 1
        assert res.duplicates_removed >= 1

        # 5. per-path ranks are within 1..3
        assert all(1 <= r <= 3 for r in res.dense_rank.values())
        assert all(1 <= r <= 3 for r in res.sparse_rank.values())

        # 6. fused_total == |dense ∪ sparse|
        dense_hits = set(res.dense_rank.keys())
        sparse_hits = set(res.sparse_rank.keys())
        assert res.fused_total == len(dense_hits | sparse_hits)

        # 7. With dedupe=False both identical chunks survive
        res_no_dedupe = retriever.search(
            query,
            mode="hybrid",
            dense_k=3,
            sparse_k=3,
            max_candidates=4,
            dedupe=False,
        )
        assert 0 in res_no_dedupe.candidates and 2 in res_no_dedupe.candidates
        assert res_no_dedupe.duplicates_removed == 0

        # 8. mode="sparse" gives an empty dense_rank
        res_sparse = retriever.search(
            query,
            mode="sparse",
            dense_k=3,
            sparse_k=3,
        )
        assert res_sparse.dense_rank == {}

        # 9. mode="dense" without dense index raises ValueError
        sparse_only_retriever = Retriever(chunks, use_dense=False)
        try:
            sparse_only_retriever.search(query, mode="dense")
            assert False, "Expected ValueError when mode='dense' without dense index"
        except ValueError:
            pass

def test_exact_vs_hnsw() -> None:
    """Verify HNSW approximate search top-10 overlap with exact cosine search >= 0.9."""
    try:
        import hnswlib
    except ImportError:
        print("Note: hnswlib not installed, skipping HNSW comparison test.")
        return

    np.random.seed(42)
    num_vecs = 500
    dim = 32
    raw = np.random.randn(num_vecs, dim).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit_vecs = raw / norms

    exact_index = DenseIndex(unit_vecs, ann_min=1_000_000)
    hnsw_index = DenseIndex(unit_vecs, ann_min=0)

    assert exact_index.kind == "exact cosine"
    assert hnsw_index.kind == "HNSW (approximate)"

    overlaps: list[float] = []
    for _ in range(20):
        q = np.random.randn(dim).astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        exact_hits = set(idx for idx, _ in exact_index.search(q, k=10))
        hnsw_hits = set(idx for idx, _ in hnsw_index.search(q, k=10))
        overlap = len(exact_hits & hnsw_hits) / 10.0
        overlaps.append(overlap)

    mean_overlap = sum(overlaps) / len(overlaps)
    assert (
        mean_overlap >= 0.9
    ), f"Mean top-10 overlap between exact and HNSW was too low: {mean_overlap:.3f}"

def test_real_corpus_sanity() -> None:
    """Verify sparse search on lecture notes finds Xavier initialization in top 3."""
    with tempfile.TemporaryDirectory() as td:
        install(td)
        chunks = load_chunks(NOTES_DIR)
        assert len(chunks) > 0, "Corpus chunks must not be empty"

        retriever = Retriever(chunks, use_dense=True)
        top3_sparse = retriever.sparse("Xavier Glorot initialisation tanh", k=3)

        # Ensure the chunk discussing Xavier initialization appears in top 3
        matching_chunks = [
            idx for idx in top3_sparse if "Xavier" in chunks[idx].text
        ]
        assert len(matching_chunks) > 0, (
            f"Expected chunk containing 'Xavier' in top 3 sparse results, "
            f"got: {[chunks[i].id for i in top3_sparse]}"
        )

def main() -> None:
    print("Running test_inverted_index_bm25_math...")
    test_inverted_index_bm25_math()

    print("Running test_analyze...")
    test_analyze()

    print("Running test_rrf...")
    test_rrf()

    print("Running test_search_shape...")
    test_search_shape()

    print("Running test_exact_vs_hnsw...")
    test_exact_vs_hnsw()

    print("Running test_real_corpus_sanity...")
    test_real_corpus_sanity()

    print("All retrieval tests passed successfully!")

if __name__ == "__main__":
    main()
