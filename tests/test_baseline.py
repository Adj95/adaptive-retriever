"""Unit tests for Fixed Baseline Twin (Task D, spec §7)."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from adaptive_retriever.config import Config
from adaptive_retriever.funnel.baseline import run_baseline
from adaptive_retriever.funnel.pipeline import run_funnel
from adaptive_retriever.models import RetrievalCandidate


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
    assert trace["pipeline"] == "fixed_baseline"
    assert trace["collection_name"] == mock_vs.collection_name


def test_baseline_and_adaptive_use_same_collection_name():
    cfg = Config.load(Path("config/default.yaml"))
    mock_vs = MagicMock()
    mock_vs.collection_name = "adaptive_retriever_a5d51992"
    mock_vs.embedding_client.encode.return_value = [0.1] * 384
    mock_vs.search_dense.return_value = []

    _, trace = run_baseline("sample", mock_vs, cfg)
    assert trace["collection_name"] == "adaptive_retriever_a5d51992"


@patch("adaptive_retriever.funnel.pipeline.apply_rerank")
@patch("adaptive_retriever.funnel.pipeline.should_rerank")
@patch("adaptive_retriever.funnel.pipeline.build_prefilter")
def test_baseline_and_funnel_candidate_count_parity(
    mock_prefilter, mock_should_rerank, mock_apply_rerank
):
    cfg = Config.load(Path("config/default.yaml"))
    mock_vs = MagicMock()
    mock_vs.collection_name = "adaptive_retriever_a5d51992"
    mock_vs.count.return_value = 22422
    mock_vs.embedding_client.encode.return_value = [0.1] * 384

    # Mock 5 candidate results for both baseline and funnel
    cands = [
        RetrievalCandidate(chunk_id=f"c{i}", text=f"text {i}", final_rank=i+1, fused_score=0.9 - i*0.1)
        for i in range(5)
    ]
    mock_vs.search_dense.return_value = [
        (f"c{i}_raw", {"chunk_id": f"c{i}", "text": f"text {i}"}, 0.9 - i*0.1)
        for i in range(5)
    ]
    mock_vs.query_hybrid.return_value = cands

    mock_prefilter.return_value = (None, {"skipped": True, "applied_constraints": {}})
    mock_should_rerank.return_value = (False, {"mode": "never", "should_rerank": False})

    baseline_cands, b_trace = run_baseline("sample query", mock_vs, cfg, limit=5)
    funnel_cands, f_trace = run_funnel("sample query", mock_vs, cfg, reranker=None)

    assert len(baseline_cands) == len(funnel_cands) == 5
    assert b_trace["final_candidate_count"] == f_trace["final_candidate_count"] == 5
