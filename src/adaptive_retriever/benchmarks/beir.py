"""BEIR Relevance Contract and Utilities (Task R-06a).

Implements is_relevant_beir using human qrels judgments.
"""

from typing import Any, Dict, Optional


def extract_doc_id(chunk: Any) -> Optional[str]:
    """Extract source document ID from a chunk object or candidate dictionary/metadata."""
    if hasattr(chunk, "doc_id") and chunk.doc_id:
        return str(chunk.doc_id)
    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        if "doc_id" in chunk.metadata:
            return str(chunk.metadata["doc_id"])
        if "_id" in chunk.metadata:
            return str(chunk.metadata["_id"])
    if isinstance(chunk, dict):
        if "doc_id" in chunk:
            return str(chunk["doc_id"])
        if "metadata" in chunk and isinstance(chunk["metadata"], dict):
            if "doc_id" in chunk["metadata"]:
                return str(chunk["metadata"]["doc_id"])
    return None


def get_qrel_score_beir(retrieved_chunk: Any, query_id: str, qrels: Dict[str, Dict[str, int]]) -> int:
    """Get the graded relevance score (e.g., 0, 1, 2) for a retrieved chunk against a query's qrels.
    
    A chunk is evaluated by its source document ID.
    Returns 0 if the doc_id is missing, query is absent, or score < 1.
    """
    doc_id = extract_doc_id(retrieved_chunk)
    if not doc_id or str(query_id) not in qrels:
        return 0
    query_qrels = qrels[str(query_id)]
    return int(query_qrels.get(doc_id, 0))


def is_relevant_beir(retrieved_chunk: Any, query_id: str, qrels: Dict[str, Dict[str, int]]) -> bool:
    """BEIR Relevance Contract (Task R-06a A3).
    
    A retrieved chunk is relevant iff its SOURCE DOCUMENT ID appears in that
    query's qrels with score >= 1.
    """
    return get_qrel_score_beir(retrieved_chunk, query_id, qrels) >= 1


def verify_beir_benchmark_harness(
    queries: list,
    qrels: Dict[str, Dict[str, int]],
    vs: Any,
    cfg: Any,
    sample_size: int = 50,
) -> None:
    """Startup Guard for BEIR Harness (Task R-06a-fix F5).
    
    Validates:
    1. Every loaded query id exists in qrels.
    2. At least 20% of queries retrieve >= 1 gold document in top-50 dense retrieval.
    """
    if not queries:
        raise ValueError("BEIR Harness Guard Failure: queries list is empty!")
    if not qrels:
        raise ValueError("BEIR Harness Guard Failure: qrels dictionary is empty!")

    # Check 1: Query ID membership in qrels
    missing_qids = [str(q["_id"]) for q in queries if str(q["_id"]) not in qrels]
    if missing_qids:
        raise AssertionError(
            f"BEIR Harness Guard Failure: {len(missing_qids)} queries missing from qrels! "
            f"Sample missing IDs: {missing_qids[:5]}"
        )

    # Check 2: At least 20% queries retrieve >= 1 gold doc in top-50 dense retrieval
    from adaptive_retriever.funnel.baseline import run_baseline
    eval_sample = queries[:sample_size]
    hits = 0
    for q in eval_sample:
        qid = str(q["_id"])
        cands, _ = run_baseline(q["text"], vs, cfg, limit=50)
        has_gold = any(is_relevant_beir(c, qid, qrels) for c in cands)
        if has_gold:
            hits += 1

    hit_rate = hits / float(len(eval_sample))
    if hit_rate < 0.20:
        raise AssertionError(
            f"BEIR Harness Guard Failure: Only {hits}/{len(eval_sample)} ({hit_rate:.1%}) queries "
            f"retrieved >= 1 gold document in top-50 dense retrieval! Required minimum is 20.0%."
        )

    print(
        f"BEIR Harness Guard PASSED: 100% of {len(queries)} query IDs present in qrels; "
        f"{hits}/{len(eval_sample)} ({hit_rate:.1%}) sample queries retrieved >= 1 gold doc in top-50."
    )
