"""Unit tests for funnel orchestrator pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from adaptive_retriever.config import Config
from adaptive_retriever.funnel.pipeline import run_funnel, warm_up
from adaptive_retriever.models import QueryClass, QueryFingerprint, RetrievalCandidate
from adaptive_retriever.vectorstore import VectorStore


@pytest.fixture
def dummy_candidates():
    return [
        RetrievalCandidate(
            chunk_id=f"c{i}",
            rep_id=f"r{i}",
            fused_score=0.016 - (i * 0.001),
            final_rank=i + 1,
            text=f"Sample text candidate {i}",
            metadata={"doc_class": "digital_prose", "doc_id": "doc123"},
        )
        for i in range(10)
    ]


def test_different_query_classes_produce_different_fusion_weights():
    cfg = Config.load(Path("config/default.yaml"))
    weights_conceptual = cfg.raw["funnel"]["hybrid"]["class_weights"]["conceptual"]
    weights_exact = cfg.raw["funnel"]["hybrid"]["class_weights"]["exact_lookup"]

    # Verify config defines different weights for different query classes
    assert weights_conceptual != weights_exact

    mock_vs = MagicMock(spec=VectorStore)
    mock_vs.count.return_value = 22422
    mock_vs.query_hybrid.return_value = []

    profile_conceptual = QueryFingerprint(
        raw_query="sample query",
        token_count=2,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.CONCEPTUAL,
        class_confidence=0.9,
    )

    profile_exact = QueryFingerprint(
        raw_query="sample query",
        token_count=2,
        has_exact_identifiers=True,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.EXACT_LOOKUP,
        class_confidence=0.9,
    )

    run_funnel("sample query", mock_vs, cfg, profile=profile_conceptual)
    assert mock_vs.query_hybrid.call_args.kwargs["query_class"] == "conceptual"

    run_funnel("sample query", mock_vs, cfg, profile=profile_exact)
    assert mock_vs.query_hybrid.call_args.kwargs["query_class"] == "exact_lookup"


@patch("adaptive_retriever.funnel.pipeline.apply_rerank")
@patch("adaptive_retriever.funnel.pipeline.should_rerank")
@patch("adaptive_retriever.funnel.pipeline.build_prefilter")
def test_run_funnel_trace_input_output_counts_and_decisions(
    mock_prefilter, mock_should_rerank, mock_apply_rerank, dummy_candidates
):
    mock_prefilter.return_value = (None, {"skipped": False, "reason": "no_indexed_constraints", "applied_constraints": {}})
    mock_should_rerank.return_value = (True, {"mode": "gated", "should_rerank": True, "margin": 0.05})
    mock_apply_rerank.return_value = dummy_candidates[:5]

    mock_vs = MagicMock(spec=VectorStore)
    mock_vs.count.return_value = 22422
    mock_vs.query_hybrid.return_value = dummy_candidates

    cfg = Config({"funnel": {"prefilter": {"enabled": True}, "hybrid": {"dense_limit": 50}, "rerank": {"mode": "gated"}}})

    candidates, trace = run_funnel(
        query_text="test query",
        vs=mock_vs,
        cfg=cfg,
        profile=None,  # Tests auto-profiling!
        reranker=MagicMock(),
    )

    assert len(candidates) == 5
    stages = trace["stages"]

    # Verify input_count is populated for every stage (no nulls!)
    assert stages["prefilter"]["input_count"] == 22422
    assert stages["prefilter"]["output_count"] == 22422
    assert stages["prefilter"]["decision"] == "no_constraints"

    assert stages["hybrid_search"]["input_count"] == 22422
    assert stages["hybrid_search"]["output_count"] == 10
    assert stages["hybrid_search"]["decision"] == "hybrid_rrf_complete"

    assert stages["rerank"]["input_count"] == 10
    assert stages["rerank"]["output_count"] == 5
    assert stages["rerank"]["decision"] == "reranked"


@patch("adaptive_retriever.funnel.pipeline.apply_rerank")
@patch("adaptive_retriever.funnel.pipeline.should_rerank")
@patch("adaptive_retriever.funnel.pipeline.build_prefilter")
def test_run_funnel_prefilter_disabled_skips_stage_1(
    mock_prefilter, mock_should_rerank, mock_apply_rerank, dummy_candidates
):
    mock_prefilter.return_value = (None, {"skipped": True, "reason": "prefilter_disabled", "applied_constraints": {}})
    mock_should_rerank.return_value = (False, {"mode": "never", "should_rerank": False})

    mock_vs = MagicMock(spec=VectorStore)
    mock_vs.count.return_value = 1000
    mock_vs.query_hybrid.return_value = dummy_candidates

    cfg = Config({"funnel": {"prefilter": {"enabled": False}, "rerank": {"mode": "never", "output_size": 5}}})

    candidates, trace = run_funnel(
        query_text="test query",
        vs=mock_vs,
        cfg=cfg,
        profile=None,
        reranker=None,
    )

    assert trace["stages"]["prefilter"]["decision"] == "skipped"
    assert trace["stages"]["prefilter"]["input_count"] == 1000
    assert trace["stages"]["rerank"]["decision"] == "skipped"
    assert len(candidates) == 5
    mock_apply_rerank.assert_not_called()


def test_warm_up_invokes_dense_sparse_and_reranker():
    mock_vs = MagicMock()
    mock_reranker = MagicMock()

    elapsed = warm_up(mock_vs, mock_reranker)
    assert isinstance(elapsed, float)
    mock_vs._ensure_dense_model.assert_called_once()
    mock_vs._ensure_sparse_model.assert_called_once()
    mock_reranker.ensure_ready.assert_called_once()
