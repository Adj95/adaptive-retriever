"""Unit tests for RRF fusion and hybrid query configuration."""

from unittest.mock import MagicMock
import pytest

from adaptive_retriever.config import Config
from adaptive_retriever.funnel.fusion import fuse_rrf
from adaptive_retriever.vectorstore import VectorStore


def test_fuse_rrf_hand_computed_fixture():
    ranked_lists = {
        "dense": ["docA", "docB"],
        "sparse": ["docB", "docC"],
    }
    rrf_k = 60
    weights = {"dense": 0.5, "sparse": 0.5}

    results = fuse_rrf(ranked_lists, rrf_k=rrf_k, weights=weights)

    assert len(results) == 3
    doc_b = results[0]
    doc_a = results[1]
    doc_c = results[2]

    # docB: 0.5 / (60 + 2) + 0.5 / (60 + 1)
    expected_b = (0.5 / 62.0) + (0.5 / 61.0)
    assert doc_b[0] == "docB"
    assert doc_b[1] == pytest.approx(expected_b)
    assert doc_b[2] == {"dense": 2, "sparse": 1}

    # docA: 0.5 / (60 + 1)
    expected_a = 0.5 / 61.0
    assert doc_a[0] == "docA"
    assert doc_a[1] == pytest.approx(expected_a)
    assert doc_a[2] == {"dense": 1, "sparse": None}

    # docC: 0.5 / (60 + 2)
    expected_c = 0.5 / 62.0
    assert doc_c[0] == "docC"
    assert doc_c[1] == pytest.approx(expected_c)
    assert doc_c[2] == {"dense": None, "sparse": 2}


def test_fuse_rrf_rrf_k_change_alters_scores():
    ranked_lists = {
        "dense": ["docA", "docB"],
        "sparse": ["docB", "docC"],
    }
    weights = {"dense": 0.5, "sparse": 0.5}

    res_60 = fuse_rrf(ranked_lists, rrf_k=60, weights=weights)
    res_1 = fuse_rrf(ranked_lists, rrf_k=1, weights=weights)

    score_a_60 = next(r[1] for r in res_60 if r[0] == "docA")
    score_a_1 = next(r[1] for r in res_1 if r[0] == "docA")

    assert score_a_60 != pytest.approx(score_a_1)
    assert score_a_1 == pytest.approx(0.5 / (1 + 1))


def test_fuse_rrf_class_weights_changes_ordering():
    ranked_lists = {
        "dense": ["docA"],
        "sparse": ["docB"],
    }

    # Equal weights: tie-break by rep_id ("docA" < "docB")
    res_equal = fuse_rrf(ranked_lists, rrf_k=60, weights={"dense": 0.5, "sparse": 0.5})
    assert [r[0] for r in res_equal] == ["docA", "docB"]

    # Sparse heavy: docB gets higher score
    res_sparse = fuse_rrf(ranked_lists, rrf_k=60, weights={"dense": 0.1, "sparse": 0.9})
    assert [r[0] for r in res_sparse] == ["docB", "docA"]


def test_fuse_rrf_single_source_rep_ranks():
    ranked_lists = {
        "dense": ["docA"],
        "sparse": ["docB"],
    }
    res = fuse_rrf(ranked_lists, rrf_k=60, weights={"dense": 0.5, "sparse": 0.5})
    map_by_doc = {r[0]: r[2] for r in res}

    assert map_by_doc["docA"] == {"dense": 1, "sparse": None}
    assert map_by_doc["docB"] == {"dense": None, "sparse": 1}


def test_query_hybrid_unknown_query_class_raises():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.0] * 384
    mock_sparse = MagicMock()
    mock_sparse.encode.return_value = MagicMock()

    vs = VectorStore.__new__(VectorStore)
    vs.embedding_client = mock_embedder
    vs.sparse_client = mock_sparse
    vs.cfg = None
    vs.search_dense = MagicMock(return_value=[])
    vs.search_sparse = MagicMock(return_value=[])

    with pytest.raises(ValueError, match="Unknown query_class 'nonexistent_class'"):
        vs.query_hybrid("test text", query_class="nonexistent_class")


def test_query_hybrid_unsupported_fusion_raises():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.0] * 384
    mock_sparse = MagicMock()
    mock_sparse.encode.return_value = MagicMock()

    vs = VectorStore.__new__(VectorStore)
    vs.embedding_client = mock_embedder
    vs.sparse_client = mock_sparse

    bad_cfg = {
        "funnel": {
            "hybrid": {
                "fusion": "linear_sum",
            }
        }
    }
    vs.cfg = bad_cfg

    with pytest.raises(ValueError, match="Unsupported fusion mode 'linear_sum'"):
        vs.query_hybrid("test text")


def test_non_default_rrf_k_and_class_weights_reaches_query_hybrid():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.0] * 384
    mock_sparse = MagicMock()
    mock_sparse.encode.return_value = MagicMock()

    vs = VectorStore.__new__(VectorStore)
    vs.embedding_client = mock_embedder
    vs.sparse_client = mock_sparse
    vs.search_dense = MagicMock(return_value=[("docA", {"text": "A", "chunk_id": "c1"}, 0.9)])
    vs.search_sparse = MagicMock(return_value=[("docB", {"text": "B", "chunk_id": "c2"}, 0.8)])

    custom_cfg = {
        "funnel": {
            "hybrid": {
                "fusion": "rrf",
                "rrf_k": 5,
                "dense_limit": 50,
                "sparse_limit": 50,
                "class_weights": {
                    "custom_class": {"dense": 0.9, "sparse": 0.1},
                    "default": {"dense": 0.5, "sparse": 0.5},
                },
            }
        }
    }

    candidates = vs.query_hybrid("query text", query_class="custom_class", cfg=custom_cfg)
    assert len(candidates) == 2

    # Verify dense_limit (50) was passed to search methods
    vs.search_dense.assert_called_once_with([0.0] * 384, limit=50, query_filter=None)
    vs.search_sparse.assert_called_once()

    # docA score: 0.9 / (5 + 1) = 0.15
    # docB score: 0.1 / (5 + 1) = 0.016666...
    assert candidates[0].rep_id == "docA"
    assert candidates[0].fused_score == pytest.approx(0.9 / 6.0)
    assert candidates[1].rep_id == "docB"
    assert candidates[1].fused_score == pytest.approx(0.1 / 6.0)
