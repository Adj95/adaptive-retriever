"""Fixed baseline twin (Task D, spec §7).

Evaluates queries using dense-only retrieval filtered strictly to kind == 'raw'
on the same Qdrant collection as the adaptive system.
No pre-filtering (beyond kind == 'raw'), no sparse BM25 search, no query
transformation, no reranking, no escalation.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client.http import models as rest

from adaptive_retriever.config import Config
from adaptive_retriever.models import RetrievalCandidate
from adaptive_retriever.vectorstore import VectorStore


def run_baseline(
    query_text: str,
    vs: VectorStore,
    cfg: Config,
    limit: Optional[int] = None,
) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
    """Run dense-only search restricted to raw representation points.

    Returns candidate list and execution trace.
    """
    t0 = time.perf_counter()
    output_limit = (
        limit
        if limit is not None
        else cfg.raw.get("funnel", {}).get("rerank", {}).get("output_size", 5)
    )

    # Filter strictly to kind == "raw" using payload index
    raw_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="kind",
                match=rest.MatchValue(value="raw"),
            )
        ]
    )

    # Ensure dense model is ready
    vs._ensure_dense_model()

    # Dense-only embedding search
    query_vector = vs.embedding_client.encode(query_text)

    raw_results = vs.search_dense(
        query_vector=query_vector,
        limit=output_limit,
        query_filter=raw_filter,
    )

    candidates: List[RetrievalCandidate] = []
    for rank, (rep_id, payload, score) in enumerate(raw_results, 1):
        chunk_id = payload.get("chunk_id", rep_id.replace("_raw", ""))
        text = payload.get("text", "")
        if not text:
            chunk = vs.get_chunk_by_rep_id(rep_id)
            if chunk:
                text = chunk.text

        cand = RetrievalCandidate(
            chunk_id=chunk_id,
            rep_id=rep_id,
            dense_score=float(score),
            sparse_score=None,
            fused_score=float(score),
            rerank_score=None,
            final_rank=rank,
            text=text,
            metadata=payload,
        )
        candidates.append(cand)

    t1 = time.perf_counter()
    elapsed_ms = round((t1 - t0) * 1000, 2)

    trace = {
        "pipeline": "fixed_baseline",
        "collection_name": vs.collection_name,
        "query_text": query_text,
        "dense_only": True,
        "kind_filter": "raw",
        "candidate_count": len(candidates),
        "elapsed_ms": elapsed_ms,
    }

    return candidates, trace
