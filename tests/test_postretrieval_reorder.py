"""Tests for post-retrieval context reordering (Task B)."""

import pytest
from adaptive_retriever.models import RetrievalCandidate
from adaptive_retriever.postretrieval.reorder import reorder_candidates


def test_reorder_set_identity_before_after():
    candidates = [
        RetrievalCandidate(chunk_id=f"c{i}", text=f"Chunk text {i}")
        for i in range(1, 6)
    ]
    cfg = {"postretrieval": {"reorder": {"enabled": True}}}
    res, trace = reorder_candidates(candidates, cfg=cfg)

    pre_ids = {c.chunk_id for c in candidates}
    post_ids = {c.chunk_id for c in res}
    assert pre_ids == post_ids
    assert len(res) == len(candidates)


def test_reorder_top_candidate_lands_first():
    candidates = [
        RetrievalCandidate(chunk_id=f"c{i}", text=f"Chunk text {i}")
        for i in range(1, 6)
    ]
    cfg = {"postretrieval": {"reorder": {"enabled": True}}}
    res, trace = reorder_candidates(candidates, cfg=cfg)

    # Top candidate (c1) MUST land at index 0 (first position)
    assert res[0].chunk_id == candidates[0].chunk_id
    # Second candidate (c2) MUST land at index N-1 (last position)
    assert res[-1].chunk_id == candidates[1].chunk_id


def test_reorder_enabled_false_order_preserving():
    candidates = [
        RetrievalCandidate(chunk_id=f"c{i}", text=f"Chunk text {i}")
        for i in range(1, 6)
    ]
    cfg = {"postretrieval": {"reorder": {"enabled": False}}}
    res, trace = reorder_candidates(candidates, cfg=cfg)

    assert [c.chunk_id for c in res] == [c.chunk_id for c in candidates]
    assert trace["enabled"] is False


def test_reorder_drop_candidate_mutation_fails():
    """Verifies that dropping any candidate causes the set identity assertion to fail."""
    candidates = [
        RetrievalCandidate(chunk_id=f"c{i}", text=f"Chunk text {i}")
        for i in range(1, 6)
    ]
    # Simulate a bug/mutation where the last candidate is dropped
    buggy_res = candidates[:-1]
    pre_ids = {c.chunk_id for c in candidates}
    post_ids = {c.chunk_id for c in buggy_res}

    with pytest.raises(AssertionError, match="Reorder set mismatch"):
        assert pre_ids == post_ids, f"Reorder set mismatch! Pre: {pre_ids}, Post: {post_ids}"
