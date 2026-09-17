"""Unit tests for Task D Risk Gate (§7)."""

import pytest
from adaptive_retriever.models import RetrievalCandidate
from adaptive_retriever.query.risk_gate import (
    compute_retrieval_confidence,
    detect_contradictions,
    evaluate_risk_gate,
)


def test_compute_retrieval_confidence_empty():
    assert compute_retrieval_confidence([]) == 0.0


def test_compute_retrieval_confidence_formula():
    cands = [
        RetrievalCandidate(chunk_id="c1", text="text1", rerank_score=0.90),
        RetrievalCandidate(chunk_id="c2", text="text2", rerank_score=0.70),
    ]
    # s1 = 0.90, s2 = 0.70, rel_margin = (0.90 - 0.70) / 0.90 = 0.2222
    # confidence = 0.90 * (0.5 + 0.5 * 0.2222) = 0.90 * 0.6111 = 0.5500
    conf = compute_retrieval_confidence(cands)
    assert conf == 0.5500


def test_detect_contradictions_date_conflict():
    cands = [
        RetrievalCandidate(chunk_id="c1", text="The policy was updated in 1998 for all users."),
        RetrievalCandidate(chunk_id="c2", text="The policy was updated in 2015 for all users."),
    ]
    res = detect_contradictions(cands)
    assert res["has_contradiction"] is True
    assert len(res["details"]) > 0


def test_detect_contradictions_no_conflict():
    cands = [
        RetrievalCandidate(chunk_id="c1", text="The feature was released in 2020."),
        RetrievalCandidate(chunk_id="c2", text="In 2020, the team finalized the feature."),
    ]
    res = detect_contradictions(cands)
    assert res["has_contradiction"] is False


def test_evaluate_risk_gate_refusal_low_confidence():
    cands = [
        RetrievalCandidate(chunk_id="c1", text="some passage", rerank_score=0.10),
    ]
    gate_res = evaluate_risk_gate("query", cands, min_confidence=0.15)
    assert gate_res.refusal_recommended is True
    assert "below minimum threshold" in gate_res.refusal_reason
