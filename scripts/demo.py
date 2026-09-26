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


DEFAULT_QUERY_ID = "PLAIN-2"
DEFAULT_QUERY_TEXT = "Do cholesterol statin drugs cause breast cancer?"
COLLECTION_NAME = "beir_nfcorpus_2e6f1c8d"


def get_resolved_device() -> str:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"cuda ({device_name})"
    return "cpu"


def check_collection_exists(qdrant_url: str, collection_name: str) -> bool:
    url_req = qdrant_url.replace("localhost", "127.0.0.1")
    try:
        with urllib.request.urlopen(f"{url_req}/collections/{collection_name}", timeout=5) as r:
            res = json.loads(r.read())
            pts = res.get("result", {}).get("points_count", 0)
            return pts > 0
    except Exception:
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
    lines.append(f"RETRIEVED top {len(top_cands)} of {len(candidates)}")
    for rank, cand in enumerate(top_cands, 1):
        # Extract doc_id from candidate metadata or chunk_id
        doc_id = cand.metadata.get("doc_id") or cand.chunk_id.split("_c")[0]
        score_str = f"score {cand.score:.3f}"
        if known_info:
            _, gold_set = known_info
            is_gold = "yes" if doc_id in gold_set or cand.chunk_id in gold_set else "no"
            lines.append(f"  {rank:<2}  {cand.chunk_id:<16} {score_str:<12}   gold: {is_gold}")
        else:
            lines.append(f"  {rank:<2}  {cand.chunk_id:<16} {score_str:<12}")

    rr_stage = trace.get("stages", {}).get("rerank", {})
    decision = rr_stage.get("decision", "skipped")
    fired = "yes" if decision == "reranked" else "no"
    gate_info = rr_stage.get("gate", {})
    margin = gate_info.get("margin", 0.0)
    thresh = gate_info.get("threshold", 0.30)
    comp = "<" if fired == "yes" else ">="
    lines.append(f"  rerank fired: {fired} (top-1 margin {margin:.2f} {comp} {thresh:.2f})")
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
    lines.append(
        f"COST      retrieval {ret_ms:,.0f} ms | rerank {rr_ms:,.0f} ms | generation {gen_ms:,.0f} ms | total {tot_ms:,.0f} ms"
    )

    return "\n".join(lines)


def run_demo(query_text: Optional[str] = None) -> str:
    print(f"Device: {get_resolved_device()}", flush=True)

    config_path = "config/beir_nfcorpus.yaml"
    if not Path(config_path).exists():
        config_path = "config/default.yaml"

    cfg = Config.load(config_path)
    qdrant_url = cfg.raw.get("vectorstore", {}).get("url", "http://localhost:6333")

    if not check_collection_exists(qdrant_url, COLLECTION_NAME):
        print(
            f"Error: Collection '{COLLECTION_NAME}' does not exist or has 0 points in Qdrant server.\n"
            f"Build it first by running: python scripts/ingest_beir_corpora.py --corpus nfcorpus",
            file=sys.stderr,
        )
        sys.exit(1)

    queries, qrels = load_nfcorpus_qrels()

    if not query_text:
        query_text = DEFAULT_QUERY_TEXT

    known_info = find_known_query_info(query_text, queries, qrels)

    vs = VectorStore.from_config(cfg, collection_base_name="beir_nfcorpus")
    llm = get_llm_client(cfg)

    # Warm up models
    _ = warm_up(vs)

    t_start = time.perf_counter()

    # Stage 1 & 2 & 3: Retrieval & Gated Rerank
    t0_ret = time.perf_counter()
    candidates, trace = run_funnel(query_text, vs, cfg, llm=llm)
    t1_ret = time.perf_counter()
    retrieval_total_ms = (t1_ret - t0_ret) * 1000.0

    stages = trace.get("stages", {})
    hy_ms = stages.get("hybrid_search", {}).get("elapsed_ms", retrieval_total_ms)
    rr_ms = stages.get("rerank", {}).get("elapsed_ms", 0.0)

    # Stage 4: Generation
    t0_gen = time.perf_counter()
    gen_ans = generate_answer(query_text, candidates, cfg=cfg, llm=llm)
    t1_gen = time.perf_counter()
    gen_ms = (t1_gen - t0_gen) * 1000.0

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000.0

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
        "total_ms": total_ms,
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
    parser.add_argument("--query", type=str, default=None, help="Query text to execute")
    args = parser.parse_args()

    output = run_demo(args.query)
    print(output)


if __name__ == "__main__":
    main()
