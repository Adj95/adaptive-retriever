"""Unit tests for gated rerank (stage 3 of retrieval funnel)."""

from unittest.mock import MagicMock, patch
import pytest

from adaptive_retriever.funnel.rerank import Reranker, apply_rerank, should_rerank
from adaptive_retriever.models import RetrievalCandidate


def test_should_rerank_confident_ranking_returns_false():
    # Realistic RRF scores: top_score = 0.016393, 5th score = 0.008196
    # Relative margin = (0.016393 - 0.008196) / 0.016393 = 0.50 >= 0.15 -> confident -> False
    scores = [0.016393, 0.014000, 0.012000, 0.010000, 0.008196, 0.007000]
    candidates = [
        RetrievalCandidate(chunk_id=f"c{i}", fused_score=s, text=f"text {i}")
        for i, s in enumerate(scores)
    ]
    cfg = {"funnel": {"rerank": {"mode": "gated", "gate_score_margin": 0.15, "output_size": 5}}}

    fire, trace = should_rerank(candidates, cfg)

    assert fire is False
    assert trace["mode"] == "gated"
    assert trace["should_rerank"] is False
    assert trace["raw_margin"] == pytest.approx(0.016393 - 0.008196)
    assert trace["margin"] == pytest.approx((0.016393 - 0.008196) / 0.016393)
    assert trace["threshold"] == 0.15


def test_should_rerank_flat_ranking_returns_true():
    # Realistic RRF scores: top_score = 0.016393, 5th score = 0.015000
    # Relative margin = (0.016393 - 0.015000) / 0.016393 = 0.08498 < 0.15 -> flat -> True
    scores = [0.016393, 0.016000, 0.015800, 0.015400, 0.015000, 0.014500]
    candidates = [
        RetrievalCandidate(chunk_id=f"c{i}", fused_score=s, text=f"text {i}")
        for i, s in enumerate(scores)
    ]
    cfg = {"funnel": {"rerank": {"mode": "gated", "gate_score_margin": 0.15, "output_size": 5}}}

    fire, trace = should_rerank(candidates, cfg)

    assert fire is True
    assert trace["should_rerank"] is True
    assert trace["margin"] == pytest.approx((0.016393 - 0.015000) / 0.016393)


def test_should_rerank_mode_always_and_never():
    candidates = [
        RetrievalCandidate(chunk_id=f"c{i}", fused_score=0.016 - (i * 0.001), text=f"text {i}")
        for i in range(10)
    ]

    fire_never, trace_never = should_rerank(candidates, cfg={"funnel": {"rerank": {"mode": "never"}}})
    assert fire_never is False
    assert trace_never["mode"] == "never"

    fire_always, trace_always = should_rerank(candidates, cfg={"funnel": {"rerank": {"mode": "always"}}})
    assert fire_always is True
    assert trace_always["mode"] == "always"


def test_should_rerank_invalid_config_raises_value_error():
    candidates = [RetrievalCandidate(chunk_id="c0", fused_score=0.01, text="text")]

    # Bare dict missing top-level "funnel"
    with pytest.raises(ValueError, match="must contain top-level 'funnel' key"):
        should_rerank(candidates, cfg={"rerank": {"mode": "gated"}})

    # Invalid type
    with pytest.raises(ValueError, match="Invalid config type"):
        should_rerank(candidates, cfg="not_a_config_object")


def test_reranker_auto_device_resolves_to_concrete_string():
    reranker = Reranker(device="auto")
    assert reranker.device in ("cuda", "cpu")
    assert reranker.device != "auto"


def test_reranker_ensure_ready_calls_ensure_loaded():
    reranker = Reranker()
    with patch.object(reranker, "_ensure_loaded") as mock_loaded:
        reranker.ensure_ready()
        mock_loaded.assert_called_once()


def test_reranker_load_failure_raises():
    with patch("sentence_transformers.CrossEncoder", side_effect=RuntimeError("Load failed")):
        reranker = Reranker(model_name="invalid-model")
        with pytest.raises(RuntimeError, match="Load failed|Failed to load reranker model"):
            reranker.score("query", ["text"])


def test_apply_rerank_length_mismatch_raises():
    mock_reranker = MagicMock(spec=Reranker)
    mock_reranker.score.return_value = [0.9]

    candidates = [
        RetrievalCandidate(chunk_id=f"c{i}", fused_score=0.01, text=f"text {i}")
        for i in range(3)
    ]

    with pytest.raises(AssertionError, match="Reranker returned 1 scores for 3 candidates"):
        apply_rerank("query", candidates, mock_reranker)


def test_apply_rerank_reorders_renumbers_and_preserves_metadata():
    mock_reranker = MagicMock(spec=Reranker)
    mock_reranker.score.return_value = [0.2, 0.9, 0.5]

    candidates = [
        RetrievalCandidate(
            chunk_id=f"c{i}",
            rep_id=f"r{i}",
            fused_score=0.016 - (i * 0.001),
            final_rank=i + 1,
            text=f"text {i}",
            metadata={"doc_id": f"doc_{i}", "custom_key": f"val_{i}"},
        )
        for i in range(3)
    ]

    cfg = {"funnel": {"rerank": {"input_size": 10, "output_size": 2}}}
    results = apply_rerank("test query", candidates, mock_reranker, cfg)

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].rerank_score == 0.9
    assert results[0].final_rank == 1
    assert results[0].metadata == {"doc_id": "doc_1", "custom_key": "val_1"}

    assert results[1].chunk_id == "c2"
    assert results[1].rerank_score == 0.5
    assert results[1].final_rank == 2
    assert results[1].metadata == {"doc_id": "doc_2", "custom_key": "val_2"}
