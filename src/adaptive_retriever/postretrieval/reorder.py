"""Post-retrieval context reordering (Task B)."""

from typing import Any, Dict, List, Optional, Tuple
from adaptive_retriever.models import RetrievalCandidate


def reorder_candidates(
    candidates: List[RetrievalCandidate],
    cfg: Optional[Any] = None,
) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
    """Reorder candidates placing highest-scoring candidates at start/end and weakest in the middle.

    Guarantees:
      - Set identity: candidate IDs set before and after reordering is identical.
      - Top candidate (pre-rank 1) always lands at position 0.
      - Deterministic for fixed input.
      - enabled: false is order-preserving pass-through.
    """
    reorder_cfg = {}
    if cfg is not None:
        if hasattr(cfg, "raw"):
            reorder_cfg = cfg.raw.get("postretrieval", {}).get("reorder", {})
        elif isinstance(cfg, dict):
            reorder_cfg = cfg.get("postretrieval", {}).get("reorder", {})

    enabled = reorder_cfg.get("enabled", True)

    if not enabled or len(candidates) <= 2:
        trace = {
            "enabled": enabled,
            "input_count": len(candidates),
            "output_count": len(candidates),
            "rank_mapping": [
                {"pre_rank": i + 1, "post_position": i, "chunk_id": c.chunk_id}
                for i, c in enumerate(candidates)
            ],
        }
        return candidates, trace

    # Lost-in-the-Middle reordering pattern:
    # Rank 1 -> pos 0, Rank 2 -> pos N-1, Rank 3 -> pos 1, Rank 4 -> pos N-2...
    n = len(candidates)
    reordered: List[Optional[RetrievalCandidate]] = [None] * n
    rank_mapping = []

    left = 0
    right = n - 1

    for i, cand in enumerate(candidates):
        pre_rank = i + 1
        if i % 2 == 0:
            pos = left
            left += 1
        else:
            pos = right
            right -= 1
        reordered[pos] = cand
        rank_mapping.append({
            "pre_rank": pre_rank,
            "post_position": pos,
            "chunk_id": cand.chunk_id,
        })

    result_candidates = [c for c in reordered if c is not None]

    # MANDATORY SET IDENTITY ASSERTION
    # Set of candidate IDs before and after must be 100% identical.
    pre_ids = {c.chunk_id for c in candidates}
    post_ids = {c.chunk_id for c in result_candidates}
    assert pre_ids == post_ids, f"Reorder set mismatch! Pre: {pre_ids}, Post: {post_ids}"

    trace = {
        "enabled": enabled,
        "input_count": len(candidates),
        "output_count": len(result_candidates),
        "rank_mapping": rank_mapping,
    }

    return result_candidates, trace
