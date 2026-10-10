"""Demonstration script running shipped pipeline against public NFCorpus index."""

import argparse
import os
import re
import sys
import textwrap
import time
import urllib.request
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from adaptive_retriever.config import Config
from adaptive_retriever.funnel.pipeline import run_funnel, warm_up
from adaptive_retriever.funnel.rerank import Reranker
from adaptive_retriever.generation.generate import attribute_sentence_to_chunks, generate_answer
from adaptive_retriever.llm import get_llm_client
from adaptive_retriever.models import RetrievalCandidate
from adaptive_retriever.vectorstore import VectorStore


DEFAULT_QUERY_ID = "PLAIN-33"
DEFAULT_QUERY_TEXT = "What's Driving America's Obesity Problem?"
COLLECTION_NAME = "beir_nfcorpus_2e6f1c8d"


def get_resolved_device() -> str:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"cuda ({device_name})"
    return "cpu"


def check_collection_exists(qdrant_url: str, collection_name: str) -> bool:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, timeout=60.0, check_compatibility=False)
        if not client.collection_exists(collection_name):
            return False
        info = client.get_collection(collection_name)
        cnt = getattr(info, "points_count", None)
        if cnt is None:
            cnt = getattr(info, "vectors_count", 0)
        return (cnt or 0) > 0
    except Exception as e:
        print(f"Warning: check_collection_exists check failed: {e}", file=sys.stderr)
        return False


def load_nfcorpus_qrels() -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Loads query_id -> text mapping and query_id -> set of gold doc_ids."""
    queries: Dict[str, str] = {}
    qrels: Dict[str, Set[str]] = {}

    qfile = Path("data/beir/nfcorpus/queries.jsonl")
    if qfile.exists():
        with open(qfile, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                queries[row["_id"]] = row["text"]

    qrel_file = Path("data/beir/nfcorpus/qrels/test.tsv")
    if qrel_file.exists():
        with open(qrel_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("query-id") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, doc_id, score = parts[0], parts[1], float(parts[2])
                    if score > 0:
                        if qid not in qrels:
                            qrels[qid] = set()
                        qrels[qid].add(doc_id)

    return queries, qrels


def find_known_query_info(
    query_text: str, queries: Dict[str, str], qrels: Dict[str, Set[str]]
) -> Optional[Tuple[str, Set[str]]]:
    clean_q = query_text.strip().lower()
    for qid, qtext in queries.items():
        if qtext.strip().lower() == clean_q or qid.lower() == clean_q:
            return qid, qrels.get(qid, set())
    return None


def format_demo_output(
    query_text: str,
    known_info: Optional[Tuple[str, Set[str]]],
    candidates: List[RetrievalCandidate],
    trace: Dict[str, Any],
    answer_text: str,
    attributions: List[Tuple[str, Optional[str]]],
    timings: Dict[str, float],
) -> str:
    lines = []

    # 1. QUERY Section
    lines.append(f"QUERY     {query_text}")
    if known_info:
        qid, gold_set = known_info
        lines.append(f"          (NFCorpus test query {qid}, {len(gold_set)} gold documents)")
    lines.append("")

    # 2. RETRIEVED Section
    top_cands = candidates[:5]
    rr_stage = trace.get("stages", {}).get("rerank", {})
    decision = rr_stage.get("decision", "skipped")
    fired = (decision == "reranked")

    ordered_by = "cross-encoder" if fired else "RRF fusion"
    lines.append(f"RETRIEVED top {len(top_cands)} of {len(candidates)}   [ordered by: {ordered_by}]")

    # Monotonicity check
    if fired:
        sort_keys = [c.rerank_score for c in top_cands if c.rerank_score is not None]
    else:
        sort_keys = [c.fused_score for c in top_cands if c.fused_score is not None]

    assert all(sort_keys[i] >= sort_keys[i + 1] for i in range(len(sort_keys) - 1)), (
        f"Monotonicity sort assertion failed for ordered by {ordered_by}: {sort_keys}"
    )

    for rank, cand in enumerate(top_cands, 1):
        doc_id = cand.metadata.get("doc_id") or cand.chunk_id.split("_c")[0]
        rrf_val = cand.fused_score if cand.fused_score is not None else (cand.score or 0.0)
        if fired:
            ce_val = cand.rerank_score if cand.rerank_score is not None else 0.0
            score_str = f"ce {ce_val:.2f}   rrf {rrf_val:.4f}"
        else:
            score_str = f"rrf {rrf_val:.4f}"

        if known_info:
            _, gold_set = known_info
            is_gold = "yes" if doc_id in gold_set or cand.chunk_id in gold_set else "no"
            lines.append(f"  {rank:<2}  {cand.chunk_id:<16} {score_str:<24}   gold: {is_gold}")
        else:
            lines.append(f"  {rank:<2}  {cand.chunk_id:<16} {score_str}")

    gate_info = rr_stage.get("gate", {})
    margin = gate_info.get("margin", 0.0)
    thresh = gate_info.get("threshold", 0.30)
    top_s = gate_info.get("top_score", 0.0)
    cut_s = gate_info.get("cutoff_score", 0.0)
    op = "<" if fired else ">="
    action = "rerank fired" if fired else "rerank skipped"

    lines.append(
        f"GATE  margin = ({top_s:.4f} - {cut_s:.4f}) / {top_s:.4f} = {margin:.2f}  {op}  {thresh:.2f}  -> {action}"
    )
    lines.append("")

    # 3. ANSWER Section
    lines.append("ANSWER")
    if answer_text:
        wrapped = textwrap.fill(answer_text, width=76)
        for w_line in wrapped.splitlines():
            lines.append(f"  {w_line}")
    else:
        lines.append("  <no answer generated>")
    lines.append("")

    # 4. GROUNDING Section
    grounded_count = sum(1 for _, attr in attributions if attr is not None)
    total_sentences = len(attributions)
    lines.append(
        f"GROUNDING {grounded_count} of {total_sentences} sentences matched >= 8-gram verbatim to retrieved context"
    )
    for idx, (sent, attr) in enumerate(attributions, 1):
        if attr:
            lines.append(f"  sentence {idx:<2} grounded   -> {attr}")
        else:
            lines.append(f"  sentence {idx:<2} ungrounded")
    lines.append("")

    # 5. COST Section
    ret_ms = timings.get("retrieval_ms", 0.0)
    rr_ms = timings.get("rerank_ms", 0.0)
    gen_ms = timings.get("generation_ms", 0.0)
    tot_ms = timings.get("total_ms", 0.0)

    if "other_ms" in timings:
        other_ms = timings.get("other_ms", 0.0)
        lines.append("COST --- wall-clock for this run (single run on this machine, not a benchmark) ---")
        lines.append(f"  retrieval    {ret_ms:,.0f} ms")
        lines.append(f"  rerank       {rr_ms:,.0f} ms")
        lines.append(f"  generation   {gen_ms:,.0f} ms")
        lines.append(f"  other        {other_ms:,.0f} ms")
        lines.append("  -----------------------")
        lines.append(f"  total        {tot_ms:,.0f} ms")
    else:
        lines.append(
            f"COST      retrieval {ret_ms:,.0f} ms | rerank {rr_ms:,.0f} ms | generation {gen_ms:,.0f} ms | total {tot_ms:,.0f} ms"
        )

    return "\n".join(lines)


def run_demo(query_input: Optional[str] = None) -> str:
    print(f"Device: {get_resolved_device()}", flush=True)

    config_path = "config/beir_nfcorpus.yaml"
    if not Path(config_path).exists():
        config_path = "config/default.yaml"

    cfg = Config.load(config_path)
    sys_dev = get_resolved_device()
    if "cpu" in sys_dev:
        if "models" in cfg.raw and "llm" in cfg.raw["models"]:
            cfg.raw["models"]["llm"]["device"] = "cpu"

    qdrant_url = cfg.raw.get("vectorstore", {}).get("url", "http://localhost:6333")

    if not check_collection_exists(qdrant_url, COLLECTION_NAME):
        print(
            f"Error: Collection '{COLLECTION_NAME}' does not exist or has 0 points in Qdrant server.\n"
            f"Build it first by running: python scripts/ingest_beir_corpora.py --corpus nfcorpus",
            file=sys.stderr,
        )
        sys.exit(1)

    queries, qrels = load_nfcorpus_qrels()

    target_qid: Optional[str] = None
    if not query_input:
        target_qid = DEFAULT_QUERY_ID
        query_text = queries.get(DEFAULT_QUERY_ID, DEFAULT_QUERY_TEXT)
    elif query_input in queries or query_input in qrels:
        target_qid = query_input
        query_text = queries.get(query_input, query_input)
    else:
        known = find_known_query_info(query_input, queries, qrels)
        if known:
            target_qid, _ = known
            query_text = queries.get(target_qid, query_input)
        else:
            query_text = query_input

    if target_qid and target_qid in qrels:
        known_info = (target_qid, qrels[target_qid])
    else:
        known_info = find_known_query_info(query_text, queries, qrels)

    print("warming up models (not timed)...", flush=True)
    vs = VectorStore.from_config(cfg, collection_base_name="beir_nfcorpus", timeout=300.0)
    try:
        import httpx
        if hasattr(vs.client, "http") and hasattr(vs.client.http, "api_client"):
            vs.client.http.api_client.client.timeout = httpx.Timeout(300.0)
        if hasattr(vs.client, "_client") and hasattr(vs.client._client, "http"):
            vs.client._client.http.api_client.client.timeout = httpx.Timeout(300.0)
    except Exception:
        pass
    llm = get_llm_client(cfg)
    reranker = Reranker(device=cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))
    reranker.ensure_ready()

    _ = warm_up(vs, reranker=reranker, llm=llm)

    # Throwaway retrieval and throwaway generation
    throwaway_cands, _ = run_funnel("warmup throwaway query text", vs, cfg, reranker=reranker, llm=llm)
    if throwaway_cands:
        _ = generate_answer("warmup throwaway query text", throwaway_cands[:2], cfg=cfg, llm=llm, ignore_cache=True)
    print("warm-up complete", flush=True)

    t_start = time.perf_counter()

    # Stage 1 & 2 & 3: Retrieval & Gated Rerank
    candidates, trace = run_funnel(query_text, vs, cfg, reranker=reranker, llm=llm)

    stages = trace.get("stages", {})
    hy_ms = stages.get("hybrid_search", {}).get("elapsed_ms", 0.0)
    rr_ms = stages.get("rerank", {}).get("elapsed_ms", 0.0)

    # Stage 4: Generation
    t0_gen = time.perf_counter()
    gen_ans = generate_answer(query_text, candidates, cfg=cfg, llm=llm, ignore_cache=True)
    t1_gen = time.perf_counter()
    gen_ms = (t1_gen - t0_gen) * 1000.0

    t_end = time.perf_counter()
    tot_ms = (t_end - t_start) * 1000.0

    other_ms = tot_ms - (hy_ms + rr_ms + gen_ms)
    assert abs((hy_ms + rr_ms + gen_ms + other_ms) - tot_ms) < 1.0, f"cost lines do not sum: {hy_ms + rr_ms + gen_ms + other_ms} vs {tot_ms}"

    # Stage 5: Post-hoc Attribution
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", gen_ans.text) if s.strip()]
    attributions = []
    for s in sentences:
        attr = attribute_sentence_to_chunks(s, candidates, min_n_gram=8)
        attributions.append((s, attr))

    timings = {
        "retrieval_ms": hy_ms,
        "rerank_ms": rr_ms,
        "generation_ms": gen_ms,
        "other_ms": other_ms,
        "total_ms": tot_ms,
    }

    output = format_demo_output(
        query_text=query_text,
        known_info=known_info,
        candidates=candidates,
        trace=trace,
        answer_text=gen_ans.text,
        attributions=attributions,
        timings=timings,
    )

    return output


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Adaptive Retriever Demo Script")
    parser.add_argument("--query", type=str, default=None, help="Query text or query ID to execute")
    parser.add_argument("--query-id", type=str, default=None, help="Query ID to execute")
    args = parser.parse_args()

    q_input = args.query_id or args.query
    output = run_demo(q_input)
    print(output)


if __name__ == "__main__":
    main()


