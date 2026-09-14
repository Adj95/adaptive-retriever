"""Reciprocal Rank Fusion over ranked lists.

Pure arithmetic: no Qdrant, no network, no models. That is deliberate - it means
fusion is unit-testable without a server, and the same two functions will later
fuse transformed query variants and IRCoT hops, not just dense and sparse.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

# (id, fused_score, {source_name: rank or None})
FusedRow = Tuple[str, float, Dict[str, Optional[int]]]


def collapse_to_best(
    ranked_ids: Sequence[str],
    group_of: Dict[str, str],
) -> Tuple[List[str], Dict[str, str]]:
    """Collapse a ranked list of member ids into a ranked list of group ids.

    Used to turn a list of representation ids into a list of chunk ids. Each
    chunk is indexed several times (raw, summary, question, keywords), so one
    chunk can occupy several positions inside a single source's ranking.

    The first (best-ranked) member of each group wins; later members of the same
    group are dropped.

    Collapsing BEFORE fusion rather than merging after it is the important part.
    If you fuse at representation level and merge afterwards, a high-tier chunk
    with four representations accumulates four RRF contributions while a
    low-tier chunk with one representation accumulates one. Chunks would then be
    ranked partly by how many representations they happen to have, which has
    nothing to do with relevance.

    Args:
        ranked_ids: member ids ordered best-first.
        group_of: member id -> group id. A member missing from the map keeps its
            own id as its group, so this degrades to the identity collapse.

    Returns:
        (group ids ordered best-first, group id -> the member that won it)
    """
    order: List[str] = []
    winner: Dict[str, str] = {}
    for member in ranked_ids:
        group = group_of.get(member, member)
        if group in winner:
            continue
        winner[group] = member
        order.append(group)
    return order, winner


def fuse_rrf(
    ranked_lists: Dict[str, List[str]],
    rrf_k: int,
    weights: Dict[str, float],
    *,
    require_weights: bool = True,
) -> List[FusedRow]:
    """Weighted Reciprocal Rank Fusion of several ordered lists.

        score(d) = sum over every source s containing d of
                       weights[s] / (rrf_k + rank_s(d))

    Rank is 1-based, so a source's top hit contributes weights[s] / (rrf_k + 1).

    Args:
        ranked_lists: source name -> ids ordered best-first. If an id appears
            more than once inside one list, its BEST (lowest) rank is used.
        rrf_k: RRF constant, must be >= 0. A larger value flattens the
            contribution curve, so agreement between sources counts for more
            than depth within any one source.
        weights: source name -> weight.
        require_weights: raise when a source has no weight. On by default on
            purpose: silently defaulting a missing source to 1.0 would give it
            twice the influence of a normal 0.5 source, and nothing would fail
            loudly enough to notice.

    Returns:
        Rows ordered by fused score descending, ties broken by id ascending so
        the output is deterministic for a fixed input.
    """
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be >= 0, got {rrf_k}")

    source_names = list(ranked_lists.keys())

    if require_weights:
        missing = [s for s in source_names if s not in weights]
        if missing:
            raise ValueError(
                f"No weight supplied for source(s) {missing}. "
                f"Weights given: {sorted(weights)}. "
                f"Refusing to guess: an unweighted source would default to 1.0 "
                f"and quietly outvote every weighted one."
            )

    ranks_by_id: Dict[str, Dict[str, int]] = {}
    for source_name, ids in ranked_lists.items():
        for rank, id_ in enumerate(ids, 1):
            per_source = ranks_by_id.setdefault(id_, {})
            if source_name not in per_source or rank < per_source[source_name]:
                per_source[source_name] = rank

    results: List[FusedRow] = []
    for id_, per_source in ranks_by_id.items():
        score = 0.0
        ranks: Dict[str, Optional[int]] = {}
        for source_name in source_names:
            rank = per_source.get(source_name)
            ranks[source_name] = rank
            if rank is not None:
                score += float(weights.get(source_name, 1.0)) / (rrf_k + rank)
        results.append((id_, score, ranks))

    results.sort(key=lambda row: (-row[1], row[0]))
    return results
