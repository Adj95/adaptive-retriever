"""Reciprocal Rank Fusion (RRF) implementation."""

from typing import Dict, List, Optional, Tuple


def fuse_rrf(
    ranked_lists: Dict[str, List[str]],
    rrf_k: int,
    weights: Dict[str, float],
) -> List[Tuple[str, float, Dict[str, Optional[int]]]]:
    """Pure Python Reciprocal Rank Fusion of multiple ordered lists of rep_ids.

    Args:
        ranked_lists: Map of source name -> list of rep_ids ordered best-first.
        rrf_k: RRF constant (e.g. 60).
        weights: Map of source name -> weight float (e.g. {"dense": 0.5, "sparse": 0.5}).

    Returns:
        Ordered list of tuples: (rep_id, fused_score, per_source_ranks), best-first.
        Ties break deterministically by rep_id ascending.
    """
    source_names = list(ranked_lists.keys())
    ranks_by_doc: Dict[str, Dict[str, int]] = {}

    for source_name, rep_ids in ranked_lists.items():
        for rank, rep_id in enumerate(rep_ids, 1):
            if rep_id not in ranks_by_doc:
                ranks_by_doc[rep_id] = {}
            ranks_by_doc[rep_id][source_name] = rank

    results: List[Tuple[str, float, Dict[str, Optional[int]]]] = []

    for rep_id, source_rank_map in ranks_by_doc.items():
        fused_score = 0.0
        per_source_ranks: Dict[str, Optional[int]] = {}
        for source_name in source_names:
            rank = source_rank_map.get(source_name)
            per_source_ranks[source_name] = rank
            if rank is not None:
                w = weights.get(source_name, 1.0)
                fused_score += w / (rrf_k + rank)
        results.append((rep_id, fused_score, per_source_ranks))

    # Sort primarily by fused_score descending, secondarily by rep_id ascending for deterministic tie-breaking.
    results.sort(key=lambda x: (-x[1], x[0]))
    return results
