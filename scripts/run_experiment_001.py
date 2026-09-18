"""Run Experiment 001: Adaptive Retriever vs Fixed Baseline Twin.

Executes both arms on the live index over the evaluation set in one warmed process.
Computes Recall@5, nDCG@5, Recall@10, nDCG@10, MRR, decomposed latencies (retrieval vs rerank p50/p95),
and bootstrapped 95% CIs for individual arm metrics and arm differences (adaptive - baseline).
Saves output to reports/experiment_001.json and prints a readable table to stdout.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np

from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import (
    bootstrap_ci,
    bootstrap_difference_ci,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from adaptive_retriever.funnel.baseline import run_baseline
from adaptive_retriever.funnel.pipeline import run_funnel, warm_up
from adaptive_retriever.funnel.rerank import Reranker
from adaptive_retriever.models import EvalItem
from adaptive_retriever.vectorstore import VectorStore


def load_eval_set(eval_path: Path) -> List[EvalItem]:
    with open(eval_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [EvalItem(**rec) for rec in records]


def main():
    t_exp_start = time.perf_counter()
    cfg = Config.load(Path("config/default.yaml"))
    seed = int(cfg.raw.get("seed", 42))
    bootstrap_samples = int(cfg.raw.get("eval", {}).get("bootstrap_samples", 1000))
    ci_level = float(cfg.raw.get("eval", {}).get("ci", 0.95))

    eval_path = Path(cfg.raw.get("paths", {}).get("eval_set", "data/eval/eval_set.jsonl"))
    all_items = load_eval_set(eval_path)

    # Provenance breakdown
    provenance_counts = {}
    for item in all_items:
        provenance_counts[item.provenance] = provenance_counts.get(item.provenance, 0) + 1

    # Filter out synthetic_rejected
    valid_items = [item for item in all_items if item.provenance != "synthetic_rejected"]
    excluded_items = [item for item in all_items if item.provenance == "synthetic_rejected"]
    exclusion_reasons = {"synthetic_rejected": len(excluded_items)}

    # Initialize shared vectorstore and reranker in ONE process
    vs = VectorStore.from_config(cfg)
    pts_count = vs.client.get_collection(vs.collection_name).points_count
    url_info = vs.url if vs.mode == "server" else str(cfg.raw.get("vectorstore", {}).get("storage_path", "data/qdrant_storage"))

    print("\n" + "=" * 80)
    print(" EXPERIMENT 001 HEADER")
    print(f"  Vectorstore Mode    : {vs.mode}")
    print(f"  Vectorstore URL/Path: {url_info}")
    print(f"  Collection Name     : {vs.collection_name}")
    print(f"  Collection Points   : {pts_count}")
    print("=" * 80 + "\n", flush=True)

    reranker = Reranker(device=cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))

    print("=== WARMING UP MODELS & QDRANT CLIENT CONNECTION ===")
    warmup_ms = warm_up(vs, reranker)
    print(f"Warm-up complete in {warmup_ms:.2f} ms | Reranker device: {reranker.get_device_str()}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    print(f"\nExecuting evaluation over {len(valid_items)} queries across both arms (depth = 10)...")

    # Data collection for baseline arm
    base_r5, base_ndcg5, base_r10, base_ndcg10, base_mrr = [], [], [], [], []
    base_ret_lat, base_rr_lat, base_tot_lat = [], [], []

    # Data collection for adaptive arm
    ad_r5, ad_ndcg5, ad_r10, ad_ndcg10, ad_mrr = [], [], [], [], []
    ad_ret_lat, ad_rr_lat, ad_tot_lat = [], [], []

    # Gate analysis collection
    gate_fired_count = 0
    gate_margins = []
    max_tokens_seen = 0

    for i, item in enumerate(valid_items, 1):
        # 1. Baseline Arm (Fixed Twin, limit=10)
        cands_base, trace_base = run_baseline(item.query, vs=vs, cfg=cfg, limit=10)
        base_texts = [c.text for c in cands_base]
        
        for txt in base_texts:
            t_cnt = len(tok.encode(txt, add_special_tokens=False))
            max_tokens_seen = max(max_tokens_seen, t_cnt)
            assert t_cnt <= 512, f"PERMANENT ASSERTION FAILED: Retrieved chunk exceeds 512 tokens ({t_cnt} tokens)!"

        base_r5.append(recall_at_k(base_texts, item.relevant_passages, k=5))
        base_ndcg5.append(ndcg_at_k(base_texts, item.relevant_passages, k=5))
        base_r10.append(recall_at_k(base_texts, item.relevant_passages, k=10))
        base_ndcg10.append(ndcg_at_k(base_texts, item.relevant_passages, k=10))
        base_mrr.append(mrr(base_texts, item.relevant_passages))

        b_ret_ms = float(trace_base.get("stages", {}).get("dense_search", {}).get("elapsed_ms", 0.0))
        base_ret_lat.append(b_ret_ms)
        base_rr_lat.append(0.0)
        base_tot_lat.append(float(trace_base.get("total_elapsed_ms", b_ret_ms)))

        # 2. Adaptive Arm (Funnel Orchestrator, output_size=10)
        cands_ad, trace_ad = run_funnel(item.query, vs=vs, cfg=cfg, reranker=reranker)
        ad_texts = [c.text for c in cands_ad]

        for txt in ad_texts:
            t_cnt = len(tok.encode(txt, add_special_tokens=False))
            max_tokens_seen = max(max_tokens_seen, t_cnt)
            assert t_cnt <= 512, f"PERMANENT ASSERTION FAILED: Retrieved chunk exceeds 512 tokens ({t_cnt} tokens)!"

        ad_r5.append(recall_at_k(ad_texts, item.relevant_passages, k=5))
        ad_ndcg5.append(ndcg_at_k(ad_texts, item.relevant_passages, k=5))
        ad_r10.append(recall_at_k(ad_texts, item.relevant_passages, k=10))
        ad_ndcg10.append(ndcg_at_k(ad_texts, item.relevant_passages, k=10))
        ad_mrr.append(mrr(ad_texts, item.relevant_passages))

        pf_ms = float(trace_ad.get("stages", {}).get("prefilter", {}).get("elapsed_ms", 0.0))
        hy_ms = float(trace_ad.get("stages", {}).get("hybrid_search", {}).get("elapsed_ms", 0.0))
        rr_ms = float(trace_ad.get("stages", {}).get("rerank", {}).get("elapsed_ms", 0.0))
        tot_ms = float(trace_ad.get("total_elapsed_ms", pf_ms + hy_ms + rr_ms))

        ad_ret_lat.append(pf_ms + hy_ms)
        ad_rr_lat.append(rr_ms)
        ad_tot_lat.append(tot_ms)

        # Gate stats
        rr_gate = trace_ad.get("stages", {}).get("rerank", {}).get("gate", {})
        if rr_gate.get("should_rerank", False):
            gate_fired_count += 1
        if "margin" in rr_gate:
            gate_margins.append(float(rr_gate["margin"]))

        if i % 25 == 0 or i == len(valid_items):
            print(f"  Processed {i}/{len(valid_items)} items...", flush=True)

    print("\n" + "="*80)
    print(f"MAX CHUNK TOKEN COUNT ASSERTION PASSED: Max token length seen across all retrieved chunks = {max_tokens_seen} <= 512")
    print("="*80 + "\n")

    # Compute bootstrapped CIs for baseline arm
    b_ci_r5 = bootstrap_ci(base_r5, n=bootstrap_samples, ci=ci_level, seed=seed)
    b_ci_ndcg5 = bootstrap_ci(base_ndcg5, n=bootstrap_samples, ci=ci_level, seed=seed)
    b_ci_r10 = bootstrap_ci(base_r10, n=bootstrap_samples, ci=ci_level, seed=seed)
    b_ci_ndcg10 = bootstrap_ci(base_ndcg10, n=bootstrap_samples, ci=ci_level, seed=seed)
    b_ci_mrr = bootstrap_ci(base_mrr, n=bootstrap_samples, ci=ci_level, seed=seed)

    # Compute bootstrapped CIs for adaptive arm
    a_ci_r5 = bootstrap_ci(ad_r5, n=bootstrap_samples, ci=ci_level, seed=seed)
    a_ci_ndcg5 = bootstrap_ci(ad_ndcg5, n=bootstrap_samples, ci=ci_level, seed=seed)
    a_ci_r10 = bootstrap_ci(ad_r10, n=bootstrap_samples, ci=ci_level, seed=seed)
    a_ci_ndcg10 = bootstrap_ci(ad_ndcg10, n=bootstrap_samples, ci=ci_level, seed=seed)
    a_ci_mrr = bootstrap_ci(ad_mrr, n=bootstrap_samples, ci=ci_level, seed=seed)

    # Compute bootstrapped CIs for differences (adaptive - baseline)
    diff_r5 = bootstrap_difference_ci(ad_r5, base_r5, n=bootstrap_samples, ci=ci_level, seed=seed)
    diff_ndcg5 = bootstrap_difference_ci(ad_ndcg5, base_ndcg5, n=bootstrap_samples, ci=ci_level, seed=seed)
    diff_r10 = bootstrap_difference_ci(ad_r10, base_r10, n=bootstrap_samples, ci=ci_level, seed=seed)
    diff_ndcg10 = bootstrap_difference_ci(ad_ndcg10, base_ndcg10, n=bootstrap_samples, ci=ci_level, seed=seed)
    diff_mrr = bootstrap_difference_ci(ad_mrr, base_mrr, n=bootstrap_samples, ci=ci_level, seed=seed)

    # Latency percentiles
    b_ret_p50, b_ret_p95 = float(np.percentile(base_ret_lat, 50)), float(np.percentile(base_ret_lat, 95))
    b_rr_p50, b_rr_p95 = 0.0, 0.0
    b_tot_p50, b_tot_p95 = float(np.percentile(base_tot_lat, 50)), float(np.percentile(base_tot_lat, 95))

    a_ret_p50, a_ret_p95 = float(np.percentile(ad_ret_lat, 50)), float(np.percentile(ad_ret_lat, 95))
    a_rr_p50, a_rr_p95 = float(np.percentile(ad_rr_lat, 50)), float(np.percentile(ad_rr_lat, 95))
    a_tot_p50, a_tot_p95 = float(np.percentile(ad_tot_lat, 50)), float(np.percentile(ad_tot_lat, 95))

    gate_skipped_count = len(valid_items) - gate_fired_count
    margin_arr = np.array(gate_margins, dtype=float) if gate_margins else np.array([0.0])

    gate_stats = {
        "total_queries": len(valid_items),
        "fired_count": gate_fired_count,
        "skipped_count": gate_skipped_count,
        "fire_rate": float(gate_fired_count / len(valid_items)) if valid_items else 0.0,
        "margin_distribution": {
            "min": float(np.min(margin_arr)),
            "p25": float(np.percentile(margin_arr, 25)),
            "p50": float(np.percentile(margin_arr, 50)),
            "p75": float(np.percentile(margin_arr, 75)),
            "p95": float(np.percentile(margin_arr, 95)),
            "max": float(np.max(margin_arr)),
            "mean": float(np.mean(margin_arr)),
            "std": float(np.std(margin_arr)),
        },
    }

    t_exp_end = time.perf_counter()
    exp_elapsed_s = round(t_exp_end - t_exp_start, 2)

    experiment_report = {
        "meta": {
            "experiment_id": "experiment_001",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "collection_name": vs.collection_name,
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "ci_level": ci_level,
            "total_items_in_file": len(all_items),
            "items_evaluated": len(valid_items),
            "items_excluded": len(excluded_items),
            "exclusion_reasons": exclusion_reasons,
            "provenance_breakdown": provenance_counts,
            "total_experiment_time_seconds": exp_elapsed_s,
            "gate_stats": gate_stats,
        },
        "baseline_fixed": {
            "recall_5": {"mean": b_ci_r5[0], "ci_95": [b_ci_r5[1], b_ci_r5[2]]},
            "ndcg_5": {"mean": b_ci_ndcg5[0], "ci_95": [b_ci_ndcg5[1], b_ci_ndcg5[2]]},
            "recall_10": {"mean": b_ci_r10[0], "ci_95": [b_ci_r10[1], b_ci_r10[2]]},
            "ndcg_10": {"mean": b_ci_ndcg10[0], "ci_95": [b_ci_ndcg10[1], b_ci_ndcg10[2]]},
            "mrr": {"mean": b_ci_mrr[0], "ci_95": [b_ci_mrr[1], b_ci_mrr[2]]},
            "retrieval_p50_ms": b_ret_p50,
            "retrieval_p95_ms": b_ret_p95,
            "rerank_p50_ms": b_rr_p50,
            "rerank_p95_ms": b_rr_p95,
            "total_p50_ms": b_tot_p50,
            "total_p95_ms": b_tot_p95,
        },
        "adaptive_retriever": {
            "recall_5": {"mean": a_ci_r5[0], "ci_95": [a_ci_r5[1], a_ci_r5[2]]},
            "ndcg_5": {"mean": a_ci_ndcg5[0], "ci_95": [a_ci_ndcg5[1], a_ci_ndcg5[2]]},
            "recall_10": {"mean": a_ci_r10[0], "ci_95": [a_ci_r10[1], a_ci_r10[2]]},
            "ndcg_10": {"mean": a_ci_ndcg10[0], "ci_95": [a_ci_ndcg10[1], a_ci_ndcg10[2]]},
            "mrr": {"mean": a_ci_mrr[0], "ci_95": [a_ci_mrr[1], a_ci_mrr[2]]},
            "retrieval_p50_ms": a_ret_p50,
            "retrieval_p95_ms": a_ret_p95,
            "rerank_p50_ms": a_rr_p50,
            "rerank_p95_ms": a_rr_p95,
            "total_p50_ms": a_tot_p50,
            "total_p95_ms": a_tot_p95,
        },
        "difference_adaptive_minus_baseline": {
            "recall_5_diff": {"mean": diff_r5[0], "ci_95": [diff_r5[1], diff_r5[2]]},
            "ndcg_5_diff": {"mean": diff_ndcg5[0], "ci_95": [diff_ndcg5[1], diff_ndcg5[2]]},
            "recall_10_diff": {"mean": diff_r10[0], "ci_95": [diff_r10[1], diff_r10[2]]},
            "ndcg_10_diff": {"mean": diff_ndcg10[0], "ci_95": [diff_ndcg10[1], diff_ndcg10[2]]},
            "mrr_diff": {"mean": diff_mrr[0], "ci_95": [diff_mrr[1], diff_mrr[2]]},
            "retrieval_p50_diff_ms": round(a_ret_p50 - b_ret_p50, 2),
            "retrieval_p95_diff_ms": round(a_ret_p95 - b_ret_p95, 2),
            "rerank_p50_diff_ms": round(a_rr_p50 - b_rr_p50, 2),
            "rerank_p95_diff_ms": round(a_rr_p95 - b_rr_p95, 2),
            "total_p50_diff_ms": round(a_tot_p50 - b_tot_p50, 2),
            "total_p95_diff_ms": round(a_tot_p95 - b_tot_p95, 2),
        },
    }

    # Save to reports/experiment_001.json
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "experiment_001.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)

    print(f"\nSaved experiment results to {out_file}")

    # Format table for stdout
    fmt_ci = lambda mean, low, high: f"{mean:.4f} [{low:.4f}, {high:.4f}]"
    fmt_diff = lambda mean, low, high: f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"

    table_lines = []
    table_lines.append("\n" + "=" * 105)
    table_lines.append(f" EXPERIMENT 001 RESULTS: ADAPTIVE RETRIEVER VS FIXED BASELINE TWIN")
    table_lines.append(f" (Seed: {seed} | Resample Iterations: {bootstrap_samples} | Confidence Level: 95%)")
    table_lines.append("=" * 105)
    table_lines.append(f"Items Evaluated: {len(valid_items)} | Items Excluded: {len(excluded_items)} ({exclusion_reasons})")
    table_lines.append(f"Provenance Breakdown: {provenance_counts}")
    table_lines.append(f"Gate Stats across {len(valid_items)} queries: Fired = {gate_fired_count} ({gate_stats['fire_rate']*100:.1f}%), Skipped = {gate_skipped_count}")
    table_lines.append(f"Relative Gate Margin (top 1 vs top 10): p50 = {gate_stats['margin_distribution']['p50']:.4f}, p95 = {gate_stats['margin_distribution']['p95']:.4f}, mean = {gate_stats['margin_distribution']['mean']:.4f}")
    if gate_fired_count == len(valid_items):
        table_lines.append("Note: The rerank gate fired on EVERY query (100.0% fire rate).")
    table_lines.append("-" * 105)
    table_lines.append(f"{'Metric':<22} | {'Fixed Baseline':<28} | {'Adaptive Retriever':<28} | {'Diff (Adaptive - Baseline) [95% CI]':<28}")
    table_lines.append("-" * 105)

    table_lines.append(f"{'Recall@5':<22} | {fmt_ci(*b_ci_r5):<28} | {fmt_ci(*a_ci_r5):<28} | {fmt_diff(*diff_r5):<28}")
    table_lines.append(f"{'nDCG@5':<22} | {fmt_ci(*b_ci_ndcg5):<28} | {fmt_ci(*a_ci_ndcg5):<28} | {fmt_diff(*diff_ndcg5):<28}")
    table_lines.append(f"{'Recall@10':<22} | {fmt_ci(*b_ci_r10):<28} | {fmt_ci(*a_ci_r10):<28} | {fmt_diff(*diff_r10):<28}")
    table_lines.append(f"{'nDCG@10':<22} | {fmt_ci(*b_ci_ndcg10):<28} | {fmt_ci(*a_ci_ndcg10):<28} | {fmt_diff(*diff_ndcg10):<28}")
    table_lines.append(f"{'MRR':<22} | {fmt_ci(*b_ci_mrr):<28} | {fmt_ci(*a_ci_mrr):<28} | {fmt_diff(*diff_mrr):<28}")
    table_lines.append("-" * 105)
    table_lines.append(f"{'Retrieval p50 (ms)':<22} | {b_ret_p50:<28.2f} | {a_ret_p50:<28.2f} | {a_ret_p50 - b_ret_p50:<+28.2f}")
    table_lines.append(f"{'Retrieval p95 (ms)':<22} | {b_ret_p95:<28.2f} | {a_ret_p95:<28.2f} | {a_ret_p95 - b_ret_p95:<+28.2f}")
    table_lines.append(f"{'Rerank p50 (ms)':<22} | {b_rr_p50:<28.2f} | {a_rr_p50:<28.2f} | {a_rr_p50 - b_rr_p50:<+28.2f}")
    table_lines.append(f"{'Rerank p95 (ms)':<22} | {b_rr_p95:<28.2f} | {a_rr_p95:<28.2f} | {a_rr_p95 - b_rr_p95:<+28.2f}")
    table_lines.append(f"{'Total Latency p50 (ms)':<22} | {b_tot_p50:<28.2f} | {a_tot_p50:<28.2f} | {a_tot_p50 - b_tot_p50:<+28.2f}")
    table_lines.append(f"{'Total Latency p95 (ms)':<22} | {b_tot_p95:<28.2f} | {a_tot_p95:<28.2f} | {a_tot_p95 - b_tot_p95:<+28.2f}")
    table_lines.append("=" * 105 + "\n")

    table_output = "\n".join(table_lines)
    print(table_output)


if __name__ == "__main__":
    main()
