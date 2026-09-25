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
