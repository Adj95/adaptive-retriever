"""Funnel orchestrator (pre-filter -> hybrid RRF -> gated rerank)."""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from adaptive_retriever.config import Config
from adaptive_retriever.funnel.prefilter import build_prefilter
from adaptive_retriever.funnel.rerank import Reranker, apply_rerank, should_rerank
from adaptive_retriever.models import RetrievalCandidate
from adaptive_retriever.query.profiler import profile_query
from adaptive_retriever.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def warm_up(vs: VectorStore, reranker: Optional[Reranker] = None) -> float:
    """Warm up embedding, sparse, Qdrant client connection, and reranker models.

    Initializes tokenizer, CUDA context, device transfers, and Qdrant gRPC/HTTP
    connections on the EXACT model instances used by the pipeline so real queries
    pay zero first-call or first-connection cold-start latencies.
    Returns elapsed warm-up time in milliseconds.
    """
    t0 = time.perf_counter()
    # 1. Warm dense embedding model
    if hasattr(vs, "embedding_client") and vs.embedding_client is not None:
        _ = vs.embedding_client.encode("warm")

    # 2. Warm sparse BM25 model
    if hasattr(vs, "sparse_client") and vs.sparse_client is not None:
        _ = vs.sparse_client.encode("warm")

    # 3. Warm full Qdrant hybrid search & raw payload retrieval connection
    if hasattr(vs, "query_hybrid"):
        try:
            cfg_to_use = getattr(vs, "cfg", None)
            if cfg_to_use is not None:
                _ = vs.query_hybrid("warmup query", limit=1, cfg=cfg_to_use)
        except Exception:
            pass

    # 4. Warm cross-encoder reranker
    if reranker is not None:
        _ = reranker.score("warm", ["warm"])

    t1 = time.perf_counter()
    warmup_ms = round((t1 - t0) * 1000, 2)
    logger.info("Pipeline warm-up complete in %.2f ms", warmup_ms)
    return warmup_ms


def run_funnel(
    query_text: str,
    vs: VectorStore,
    cfg: Config,
    profile: Optional[Any] = None,
    reranker: Optional[Reranker] = None,
) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
    """pre-filter -> hybrid RRF -> gated rerank. Returns candidates and a trace."""
    t_start = time.perf_counter()

    # 0. Query Profiling (Correction 1: wire profiler when profile is None)
    if profile is None:
        profile = profile_query(query_text)

    query_cls = None
    class_confidence = 1.0
    if profile is not None:
        if hasattr(profile, "query_class"):
            qc = profile.query_class
            query_cls = qc.value if hasattr(qc, "value") else str(qc)
        elif isinstance(profile, dict) and "query_class" in profile:
            query_cls = str(profile["query_class"])

        if hasattr(profile, "class_confidence"):
            class_confidence = float(profile.class_confidence)
        elif isinstance(profile, dict) and "class_confidence" in profile:
            class_confidence = float(profile["class_confidence"])

    # Stage 1: Structured Pre-filter
    total_collection_points = 0
    try:
        res = vs.count()
        if isinstance(res, int):
            total_collection_points = res
    except Exception:
        total_collection_points = 0

    t0_pf = time.perf_counter()
    filter_obj, pf_trace = build_prefilter(profile, cfg=cfg)

    if pf_trace.get("skipped"):
        pf_decision = "skipped"
        pf_out_count = total_collection_points
    elif filter_obj is None:
        pf_decision = "no_constraints"
        pf_out_count = total_collection_points
    else:
        pf_decision = "applied"
        try:
            res_pf = vs.count(count_filter=filter_obj)
            pf_out_count = res_pf if isinstance(res_pf, int) else total_collection_points
        except Exception:
            pf_out_count = total_collection_points

    t1_pf = time.perf_counter()
    pf_ms = round((t1_pf - t0_pf) * 1000, 2)

    pf_stage_trace = {
        "input_count": total_collection_points,
        "output_count": pf_out_count,
        "elapsed_ms": pf_ms,
        "decision": pf_decision,
        "details": pf_trace,
    }

    # Stage 2: Hybrid RRF Search (passing query_class for per-class weights)
    hybrid_limit = cfg.raw.get("funnel", {}).get("hybrid", {}).get("dense_limit", 100)
    t0_hy = time.perf_counter()
    hybrid_candidates = vs.query_hybrid(
        query_text=query_text,
        limit=hybrid_limit,
        doc_class_filter=None,
        query_class=query_cls,
        cfg=cfg,
        query_filter=filter_obj,
    )
    t1_hy = time.perf_counter()
    hy_ms = round((t1_hy - t0_hy) * 1000, 2)

    # Extract step breakdown from vs trace if available
    hy_details = getattr(vs, "last_query_breakdown", {})
    if not isinstance(hy_details, dict):
        hy_details = {}

    hy_stage_trace = {
        "input_count": pf_out_count,
        "output_count": len(hybrid_candidates),
        "elapsed_ms": hy_ms,
        "decision": "hybrid_rrf_complete",
        "query_class": query_cls,
        "class_confidence": class_confidence,
        "step_breakdown_ms": hy_details,
    }

    # Stage 3: Gated Rerank
    t0_rr = time.perf_counter()
    fire_rerank, rerank_gate_trace = should_rerank(hybrid_candidates, cfg=cfg)

    if fire_rerank:
        active_reranker = reranker
        if active_reranker is None:
            llm_device = cfg.raw.get("models", {}).get("llm", {}).get("device", "auto")
            active_reranker = Reranker(device=llm_device)
            active_reranker.ensure_ready()
        final_candidates = apply_rerank(query_text, hybrid_candidates, active_reranker, cfg=cfg)
        rerank_decision = "reranked"
        rerank_details = getattr(active_reranker, "last_rerank_breakdown", {})
        if not isinstance(rerank_details, dict):
            rerank_details = {}
    else:
        output_sz = cfg.raw.get("funnel", {}).get("rerank", {}).get("output_size", 5)
        top_candidates = hybrid_candidates[:output_sz]
        final_candidates = []
        for rank, cand in enumerate(top_candidates, 1):
            renumbered = cand.model_copy(update={"final_rank": rank})
            final_candidates.append(renumbered)
        rerank_decision = "skipped"
        rerank_details = {}

    t1_rr = time.perf_counter()
    rr_ms = round((t1_rr - t0_rr) * 1000, 2)

    rr_stage_trace = {
        "input_count": len(hybrid_candidates),
        "output_count": len(final_candidates),
        "elapsed_ms": rr_ms,
        "decision": rerank_decision,
        "gate": rerank_gate_trace,
        "step_breakdown_ms": rerank_details,
    }

    t_end = time.perf_counter()
    total_ms = round((t_end - t_start) * 1000, 2)
    trace_id = hashlib.sha256(f"{query_text}_{t_start}".encode("utf-8")).hexdigest()[:16]

    full_trace = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": {
            "raw_query": query_text,
            "query_class": query_cls,
            "class_confidence": class_confidence,
        },
        "stages": {
            "prefilter": pf_stage_trace,
            "hybrid_search": hy_stage_trace,
            "rerank": rr_stage_trace,
        },
        "total_elapsed_ms": total_ms,
        "final_candidate_count": len(final_candidates),
    }

    # Record trace to reports/traces/traces.jsonl
    reports_dir = Path("reports/traces")
    reports_dir.mkdir(parents=True, exist_ok=True)
    trace_file = reports_dir / "traces.jsonl"
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(full_trace) + "\n")

    return final_candidates, full_trace
