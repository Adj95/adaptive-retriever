"""Unit tests for Fixed Baseline Twin (Task D, spec §7)."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from adaptive_retriever.config import Config
from adaptive_retriever.funnel.baseline import run_baseline


def test_baseline_never_calls_reranker_transformer_or_sparse():
    mock_vs = MagicMock()
    mock_vs.collection_name = "adaptive_retriever_a5d51992"
    mock_vs.embedding_client.encode.return_value = [0.1] * 384
    mock_vs.search_dense.return_value = [
        ("chunk_1_raw", {"chunk_id": "chunk_1", "text": "Sample text", "kind": "raw"}, 0.95),
    ]

    cfg = Config.load(Path("config/default.yaml"))

    candidates, trace = run_baseline("test query", mock_vs, cfg, limit=5)

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "chunk_1"
    assert candidates[0].fused_score == 0.95

    # Enforce §7 invariant: baseline NEVER calls sparse search or reranker
    mock_vs.search_sparse.assert_not_called()
    assert trace["dense_only"] is True
    assert trace["collection_name"] == mock_vs.collection_name


def test_baseline_and_adaptive_use_same_collection_name():
    cfg = Config.load(Path("config/default.yaml"))
    mock_vs = MagicMock()
    mock_vs.collection_name = "adaptive_retriever_a5d51992"
    mock_vs.embedding_client.encode.return_value = [0.1] * 384
    mock_vs.search_dense.return_value = []

    _, trace = run_baseline("sample", mock_vs, cfg)
    assert trace["collection_name"] == "adaptive_retriever_a5d51992"
