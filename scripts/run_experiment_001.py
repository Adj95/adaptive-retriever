"""Run Experiment 001: Adaptive Retriever vs Fixed Baseline Twin.

Executes both arms on the live index over the evaluation set in one warmed process.
Computes Recall@5, Recall@10, nDCG@10, MRR, latency p50/p95, and bootstrapped 95% CIs
for individual arm metrics and arm differences (adaptive - baseline).
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
    normalise,
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
    reranker = Reranker(device=cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))

    print("=== WARMING UP MODELS & QDRANT CLIENT CONNECTION ===")
    warmup_ms = warm_up(vs, reranker)
    print(f"Warm-up complete in {warmup_ms:.2f} ms | Reranker device: {reranker.get_device_str()}")

    print(f"\nExecuting evaluation over {len(valid_items)} queries across both arms...")

    # Data collection for baseline arm
    base_r5, base_r10, base_ndcg10, base_mrr, base_lat = [], [], [], [], []
    # Data collection for adaptive arm
    ad_r5, ad_r10, ad_ndcg10, ad_mrr, ad_lat = [], [], [], [], []

    for i, item in enumerate(valid_items, 1):
        # 1. Baseline Arm (Fixed Twin)
        cands_base, trace_base = run_baseline(item.query, vs=vs, cfg=cfg, limit=5)
        base_texts = [c.text for c in cands_base]
        base_r5.append(recall_at_k(base_texts, item.relevant_passages, k=5))
        base_r10.append(recall_at_k(base_texts, item.relevant_passages, k=10))
        base_ndcg10.append(ndcg_at_k(base_texts, item.relevant_passages, k=10))
        base_mrr.append(mrr(base_texts, item.relevant_passages))
        base_lat.append(float(trace_base.get("total_elapsed_ms", 0.0)))

        # 2. Adaptive Arm (Funnel Orchestrator)
        cands_ad, trace_ad = run_funnel(item.query, vs=vs, cfg=cfg, reranker=reranker)
        ad_texts = [c.text for c in cands_ad]
        ad_r5.append(recall_at_k(ad_texts, item.relevant_passages, k=5))
        ad_r10.append(recall_at_k(ad_texts, item.relevant_passages, k=10))
        ad_ndcg10.append(ndcg_at_k(ad_texts, item.relevant_passages, k=10))
        ad_mrr.append(mrr(ad_texts, item.relevant_passages))
        ad_lat.append(float(trace_ad.get("total_elapsed_ms", 0.0)))

        if i % 25 == 0 or i == len(valid_items):
            print(f"  Processed {i}/{len(valid_items)} items...")

    # Compute bootstrapped CIs for baseline arm
    b_ci_r5 = bootstrap_ci(base_r5, n=bootstrap_samples, ci=ci_level, seed=seed)
    b_ci_r10 = bootstrap_ci(base_r10, n=bootstrap_samples, ci=ci_level, seed=seed)
    b_ci_ndcg = bootstrap_ci(base_ndcg10, n=bootstrap_samples, ci=ci_level, seed=seed)
    b_ci_mrr = bootstrap_ci(base_mrr, n=bootstrap_samples, ci=ci_level, seed=seed)

    # Compute bootstrapped CIs for adaptive arm
    a_ci_r5 = bootstrap_ci(ad_r5, n=bootstrap_samples, ci=ci_level, seed=seed)
    a_ci_r10 = bootstrap_ci(ad_r10, n=bootstrap_samples, ci=ci_level, seed=seed)
    a_ci_ndcg = bootstrap_ci(ad_ndcg10, n=bootstrap_samples, ci=ci_level, seed=seed)
    a_ci_mrr = bootstrap_ci(ad_mrr, n=bootstrap_samples, ci=ci_level, seed=seed)

    # Compute bootstrapped CIs for differences (adaptive - baseline)
    diff_r5 = bootstrap_difference_ci(ad_r5, base_r5, n=bootstrap_samples, ci=ci_level, seed=seed)
    diff_r10 = bootstrap_difference_ci(ad_r10, base_r10, n=bootstrap_samples, ci=ci_level, seed=seed)
    diff_ndcg = bootstrap_difference_ci(ad_ndcg10, base_ndcg10, n=bootstrap_samples, ci=ci_level, seed=seed)
    diff_mrr = bootstrap_difference_ci(ad_mrr, base_mrr, n=bootstrap_samples, ci=ci_level, seed=seed)

    base_lat_p50 = float(np.percentile(base_lat, 50))
    base_lat_p95 = float(np.percentile(base_lat, 95))

    ad_lat_p50 = float(np.percentile(ad_lat, 50))
    ad_lat_p95 = float(np.percentile(ad_lat, 95))

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
        },
        "baseline_fixed": {
            "recall_5": {"mean": b_ci_r5[0], "ci_95": [b_ci_r5[1], b_ci_r5[2]]},
            "recall_10": {"mean": b_ci_r10[0], "ci_95": [b_ci_r10[1], b_ci_r10[2]]},
            "ndcg_10": {"mean": b_ci_ndcg[0], "ci_95": [b_ci_ndcg[1], b_ci_ndcg[2]]},
            "mrr": {"mean": b_ci_mrr[0], "ci_95": [b_ci_mrr[1], b_ci_mrr[2]]},
            "latency_p50_ms": base_lat_p50,
            "latency_p95_ms": base_lat_p95,
        },
        "adaptive_retriever": {
            "recall_5": {"mean": a_ci_r5[0], "ci_95": [a_ci_r5[1], a_ci_r5[2]]},
            "recall_10": {"mean": a_ci_r10[0], "ci_95": [a_ci_r10[1], a_ci_r10[2]]},
            "ndcg_10": {"mean": a_ci_ndcg[0], "ci_95": [a_ci_ndcg[1], a_ci_ndcg[2]]},
            "mrr": {"mean": a_ci_mrr[0], "ci_95": [a_ci_mrr[1], a_ci_mrr[2]]},
            "latency_p50_ms": ad_lat_p50,
            "latency_p95_ms": ad_lat_p95,
        },
        "difference_adaptive_minus_baseline": {
            "recall_5_diff": {"mean": diff_r5[0], "ci_95": [diff_r5[1], diff_r5[2]]},
            "recall_10_diff": {"mean": diff_r10[0], "ci_95": [diff_r10[1], diff_r10[2]]},
            "ndcg_10_diff": {"mean": diff_ndcg[0], "ci_95": [diff_ndcg[1], diff_ndcg[2]]},
            "mrr_diff": {"mean": diff_mrr[0], "ci_95": [diff_mrr[1], diff_mrr[2]]},
            "latency_p50_diff_ms": round(ad_lat_p50 - base_lat_p50, 2),
            "latency_p95_diff_ms": round(ad_lat_p95 - base_lat_p95, 2),
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
    table_lines.append("-" * 105)
    table_lines.append(f"{'Metric':<18} | {'Fixed Baseline':<30} | {'Adaptive Retriever':<30} | {'Diff (Adaptive - Baseline) [95% CI]':<30}")
    table_lines.append("-" * 105)

    table_lines.append(f"{'Recall@5':<18} | {fmt_ci(*b_ci_r5):<30} | {fmt_ci(*a_ci_r5):<30} | {fmt_diff(*diff_r5):<30}")
    table_lines.append(f"{'Recall@10':<18} | {fmt_ci(*b_ci_r10):<30} | {fmt_ci(*a_ci_r10):<30} | {fmt_diff(*diff_r10):<30}")
    table_lines.append(f"{'nDCG@10':<18} | {fmt_ci(*b_ci_ndcg):<30} | {fmt_ci(*a_ci_ndcg):<30} | {fmt_diff(*diff_ndcg):<30}")
    table_lines.append(f"{'MRR':<18} | {fmt_ci(*b_ci_mrr):<30} | {fmt_ci(*a_ci_mrr):<30} | {fmt_diff(*diff_mrr):<30}")
    table_lines.append(f"{'Latency p50 (ms)':<18} | {base_lat_p50:<30.2f} | {ad_lat_p50:<30.2f} | {ad_lat_p50 - base_lat_p50:<+30.2f}")
    table_lines.append(f"{'Latency p95 (ms)':<18} | {base_lat_p95:<30.2f} | {ad_lat_p95:<30.2f} | {ad_lat_p95 - base_lat_p95:<+30.2f}")
    table_lines.append("=" * 105 + "\n")

    table_output = "\n".join(table_lines)
    print(table_output)


if __name__ == "__main__":
    main()
