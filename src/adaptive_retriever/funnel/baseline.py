"""Fixed baseline twin (Task D, spec §7).

Evaluates queries using dense-only retrieval filtered strictly to kind == 'raw'
on the same Qdrant collection as the adaptive system.
No pre-filtering (beyond kind == 'raw'), no sparse BM25 search, no query
transformation, no reranking, no escalation.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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

    Returns candidate list and execution trace matching run_funnel format.
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
    if hasattr(vs, "embedding_client") and hasattr(vs.embedding_client, "_ensure_loaded"):
        vs.embedding_client._ensure_loaded()

    # Total points in collection for trace input_count parity
    total_collection_points = 0
    try:
        res = vs.count()
        if isinstance(res, int):
            total_collection_points = res
    except Exception:
        total_collection_points = 0

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
    trace_id = hashlib.sha256(f"baseline_{query_text}_{t0}".encode("utf-8")).hexdigest()[:16]

    trace = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": "fixed_baseline",
        "collection_name": vs.collection_name,
        "query": {
            "raw_query": query_text,
            "query_class": "raw_baseline",
        },
        "stages": {
            "dense_search": {
                "input_count": total_collection_points,
                "output_count": len(candidates),
                "elapsed_ms": elapsed_ms,
                "decision": "dense_raw_complete",
                "kind_filter": "raw",
            }
        },
        "total_elapsed_ms": elapsed_ms,
        "final_candidate_count": len(candidates),
    }

    # Record trace to reports/traces/traces.jsonl
    reports_dir = Path("reports/traces")
    reports_dir.mkdir(parents=True, exist_ok=True)
    trace_file = reports_dir / "traces.jsonl"
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")

    return candidates, trace
