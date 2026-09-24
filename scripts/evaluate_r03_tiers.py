#!/usr/bin/env python3
"""Task R-03: Tier Definitions, Cost, and Quality Evaluation.

Evaluates Tiers 1, 2, 3 across all N=169 validation queries.
Saves report to reports/r03_tiers.json.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import mrr, ndcg_at_k, recall_at_k
from adaptive_retriever.funnel.pipeline import run_funnel, warm_up
from adaptive_retriever.funnel.rerank import Reranker
from adaptive_retriever.models import EvalItem
from adaptive_retriever.vectorstore import VectorStore

logging.basicConfig(level=logging.WARNING)

EVAL_PATH = Path("data/eval/eval_set.jsonl")
REPORT_PATH = Path("reports/r03_tiers.json")

def print_header(vs: VectorStore, reranker: Reranker):
    print("=" * 80)
    print("TASK R-03: EVALUATING RETRIEVAL TIERS 1, 2, 3")
    print(f"Python Executable: {sys.executable}")
    import torch
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    print(f"CUDA Version: {cuda_ver}")
    print(f"Resolved Reranker Device: {reranker.get_device_str()}")
    print("=" * 80, flush=True)

def build_tier_config(cfg: Config, tier_num: int) -> Config:
    import copy
    new_raw = copy.deepcopy(cfg.raw)
    
    if tier_num == 1:
        # Tier 1: dense only, k=5, rerank OFF (never), kind=raw only, top-3 to generator
        new_raw.setdefault("funnel", {}).setdefault("hybrid", {})["dense_limit"] = 5
        new_raw["funnel"]["hybrid"]["sparse_limit"] = 0
        new_raw["funnel"]["hybrid"]["weights"] = {"dense": 1.0, "sparse": 0.0}
        new_raw.setdefault("funnel", {}).setdefault("prefilter", {})["enabled"] = True
        new_raw["funnel"]["prefilter"]["kind_filter"] = "raw"
        new_raw.setdefault("funnel", {}).setdefault("rerank", {})["mode"] = "never"
        new_raw["funnel"]["rerank"]["output_size"] = 5
        new_raw.setdefault("generation", {})["context_top_k"] = 3
    elif tier_num == 2:
        # Tier 2: hybrid dense+sparse, k=10, rerank ON (always), kind=raw only, top-5 to generator
        new_raw.setdefault("funnel", {}).setdefault("hybrid", {})["dense_limit"] = 10
        new_raw["funnel"]["hybrid"]["sparse_limit"] = 10
        new_raw["funnel"]["hybrid"]["weights"] = {"dense": 0.5, "sparse": 0.5}
        new_raw.setdefault("funnel", {}).setdefault("prefilter", {})["enabled"] = True
        new_raw["funnel"]["prefilter"]["kind_filter"] = "raw"
        new_raw.setdefault("funnel", {}).setdefault("rerank", {})["mode"] = "always"
        new_raw["funnel"]["rerank"]["input_size"] = 10
        new_raw["funnel"]["rerank"]["output_size"] = 10
        new_raw.setdefault("generation", {})["context_top_k"] = 5
    elif tier_num == 3:
        # Tier 3: hybrid dense+sparse, k=20, rerank ON (always), all kinds, top-10 to generator
        new_raw.setdefault("funnel", {}).setdefault("hybrid", {})["dense_limit"] = 20
        new_raw["funnel"]["hybrid"]["sparse_limit"] = 20
        new_raw["funnel"]["hybrid"]["weights"] = {"dense": 0.5, "sparse": 0.5}
        new_raw.setdefault("funnel", {}).setdefault("prefilter", {})["enabled"] = True
        new_raw["funnel"]["prefilter"]["kind_filter"] = "all"
        new_raw.setdefault("funnel", {}).setdefault("rerank", {})["mode"] = "always"
        new_raw["funnel"]["rerank"]["input_size"] = 20
        new_raw["funnel"]["rerank"]["output_size"] = 20
        new_raw.setdefault("generation", {})["context_top_k"] = 10
    else:
        raise ValueError(f"Unknown tier_num: {tier_num}")
        
    return Config(new_raw)

def compute_bootstrap_pvalue(diffs: np.ndarray) -> float:
    obs_mean = np.mean(diffs)
    if obs_mean == 0:
        return 1.0
    centered = diffs - obs_mean
    if obs_mean > 0:
        p_val = np.mean(centered >= obs_mean) * 2.0
    else:
        p_val = np.mean(centered <= obs_mean) * 2.0
    return min(1.0, max(0.0, float(p_val)))

def evaluate_r03():
    base_cfg = Config.load("config/default.yaml")
    if "models" in base_cfg.raw and "llm" in base_cfg.raw["models"]:
        base_cfg.raw["models"]["llm"]["device"] = "auto"

    vs = VectorStore.from_config(base_cfg)
    reranker = Reranker(
        model_name=base_cfg.raw.get("models", {}).get("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        device="auto",
    )
    
    print_header(vs, reranker)
    warmup_ms = warm_up(vs, reranker)
    print(f"Cold Start Warmup Time: {warmup_ms:.2f} ms (EXCLUDED from stage latency numbers)\n", flush=True)

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        all_eval_items = [EvalItem(**json.loads(line)) for line in f if line.strip()]

    valid_eval_items = [item for item in all_eval_items if item.provenance != "synthetic_rejected"]
    N = len(valid_eval_items)
    print(f"Loaded {N} valid evaluation queries (N=169).", flush=True)

    tier_configs = {t: build_tier_config(base_cfg, t) for t in (1, 2, 3)}
    
    # Storage for tier evaluation results
    # per item per tier: cost_ms, cost_tokens, candidates, metrics
    tier_results = {1: [], 2: [], 3: []}

    for t_num in (1, 2, 3):
        t_cfg = tier_configs[t_num]
        print(f"\nEvaluating Tier {t_num}...", flush=True)
        
        for i, item in enumerate(valid_eval_items, 1):
            query = item.query
            gold_passages = item.relevant_passages
            
            # Wall clock start
            t0 = time.perf_counter()
            cands, trace = run_funnel(query, vs=vs, cfg=t_cfg, reranker=reranker, llm=None)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            
            # Count context tokens fed to generator
            context_k = t_cfg.raw.get("generation", {}).get("context_top_k", 3)
            context_cands = cands[:context_k]
            total_context_chars = sum(len(c.text) for c in context_cands)
            approx_tokens = int(len(query.split()) * 1.3) + int(total_context_chars / 4.0)
            
            texts = [c.text for c in cands]
            r5 = recall_at_k(texts, gold_passages, k=5)
            r10 = recall_at_k(texts, gold_passages, k=10)
            ndcg5 = ndcg_at_k(texts, gold_passages, k=5)
            ndcg10 = ndcg_at_k(texts, gold_passages, k=10)
            mrr_val = mrr(texts, gold_passages)
            
            tier_results[t_num].append({
                "item_id": getattr(item, "item_id", str(i)),
                "elapsed_ms": elapsed_ms,
                "tokens": approx_tokens,
                "r5": r5,
                "r10": r10,
                "ndcg5": ndcg5,
                "ndcg10": ndcg10,
                "mrr": mrr_val,
            })

    # Compute p50 and p95 cost for each tier
    tier1_p50_ms = float(np.median([r["elapsed_ms"] for r in tier_results[1]]))
    
    cost_summary = {}
    for t_num in (1, 2, 3):
        ms_vals = [r["elapsed_ms"] for r in tier_results[t_num]]
        tok_vals = [r["tokens"] for r in tier_results[t_num]]
        
        # cost_units normalized to Tier 1 p50 ms
        unit_vals = [m / tier1_p50_ms for m in ms_vals]
        
        for r, u in zip(tier_results[t_num], unit_vals):
            r["cost_units"] = u

        cost_summary[f"tier_{t_num}"] = {
            "cost_ms": {
                "p50": float(np.percentile(ms_vals, 50)),
                "p95": float(np.percentile(ms_vals, 95)),
            },
            "cost_tokens": {
                "p50": float(np.percentile(tok_vals, 50)),
                "p95": float(np.percentile(tok_vals, 95)),
            },
            "cost_units": {
                "p50": float(np.percentile(unit_vals, 50)),
                "p95": float(np.percentile(unit_vals, 95)),
            }
        }

    # Cost spread ratio
    t1_units_p50 = cost_summary["tier_1"]["cost_units"]["p50"]
    t3_units_p50 = cost_summary["tier_3"]["cost_units"]["p50"]
    cost_spread_ratio = t3_units_p50 / t1_units_p50 if t1_units_p50 > 0 else 1.0

    print("\n" + "=" * 80)
    print("COST SUMMARY PER TIER")
    print(json.dumps(cost_summary, indent=2))
    print(f"Cost Spread Ratio (Tier 3 p50 units / Tier 1 p50 units): {cost_spread_ratio:.2f}x")
    print("=" * 80 + "\n", flush=True)

    # Paired Bootstrap Quality Comparisons (10,000 resamples)
    np.random.seed(42)
    bootstrap_samples = 10000
    metric_keys = ["r5", "ndcg5", "r10", "ndcg10", "mrr"]
    metric_labels = ["Recall@5", "nDCG@5", "Recall@10", "nDCG@10", "MRR"]
    alpha_bonferroni = 0.0050 # 5 metrics x 2 comparisons

    quality_table = []
    
    # Calculate means per tier
    tier_means = {}
    for t_num in (1, 2, 3):
        tier_means[t_num] = {
            k: float(np.mean([r[k] for r in tier_results[t_num]])) for k in metric_keys
        }

    # Resample indices
    resample_idx = np.random.randint(0, N, size=(bootstrap_samples, N))

    for k, label in zip(metric_keys, metric_labels):
        m1 = tier_means[1][k]
        m2 = tier_means[2][k]
        m3 = tier_means[3][k]
        
        arr1 = np.array([r[k] for r in tier_results[1]])
        arr2 = np.array([r[k] for r in tier_results[2]])
        arr3 = np.array([r[k] for r in tier_results[3]])

        # Bootstrap T1, T2, T3
        bs_t1 = np.mean(arr1[resample_idx], axis=1)
        bs_t2 = np.mean(arr2[resample_idx], axis=1)
        bs_t3 = np.mean(arr3[resample_idx], axis=1)

        diff21 = bs_t2 - bs_t1
        diff31 = bs_t3 - bs_t1

        diff21_mean = float(m2 - m1)
        diff31_mean = float(m3 - m1)

        p21 = compute_bootstrap_pvalue(diff21)
        p31 = compute_bootstrap_pvalue(diff31)

        ci21 = [float(np.percentile(diff21, 2.5)), float(np.percentile(diff21, 97.5))]
        ci31 = [float(np.percentile(diff31, 2.5)), float(np.percentile(diff31, 97.5))]

        pass21 = "PASS" if (p21 < alpha_bonferroni and diff21_mean > 0) else "FAIL"
        pass31 = "PASS" if (p31 < alpha_bonferroni and diff31_mean > 0) else "FAIL"

        quality_table.append({
            "metric": label,
            "Tier_1": f"{m1:.4f} [{np.percentile(bs_t1, 2.5):.4f}, {np.percentile(bs_t1, 97.5):.4f}]",
            "Tier_2": f"{m2:.4f} [{np.percentile(bs_t2, 2.5):.4f}, {np.percentile(bs_t2, 97.5):.4f}]",
            "Tier_3": f"{m3:.4f} [{np.percentile(bs_t3, 2.5):.4f}, {np.percentile(bs_t3, 97.5):.4f}]",
            "T2_vs_T1_diff": f"{diff21_mean:+.4f} [{ci21[0]:+.4f}, {ci21[1]:+.4f}] (p={p21:.4f})",
            "T2_vs_T1_status": pass21,
            "T3_vs_T1_diff": f"{diff31_mean:+.4f} [{ci31[0]:+.4f}, {ci31[1]:+.4f}] (p={p31:.4f})",
            "T3_vs_T1_status": pass31,
            "mean_Tier1": m1,
            "mean_Tier2": m2,
            "mean_Tier3": m3,
            "diff_T2_T1": diff21_mean,
            "diff_T3_T1": diff31_mean,
            "p_T2_T1": p21,
            "p_T3_T1": p31,
        })

    print("=" * 110)
    print("QUALITY SUMMARY PER TIER")
    print(f"{'Metric':<10} | {'Tier 1 (Dense k=5)':<20} | {'Tier 2 (Hybrid k=10)':<20} | {'Tier 3 (Multi-Rep k=20)':<22} | {'T2 vs T1 Diff':<18} | {'T3 vs T1 Diff':<18}")
    print("-" * 110)
    for row in quality_table:
        print(f"{row['metric']:<10} | {row['Tier_1']:<20} | {row['Tier_2']:<20} | {row['Tier_3']:<22} | {row['T2_vs_T1_diff']:<18} | {row['T3_vs_T1_diff']:<18}")
    print("=" * 110 + "\n", flush=True)

    report_output = {
        "warmup_ms": warmup_ms,
        "total_queries": N,
        "cost_spread_ratio_p50": cost_spread_ratio,
        "cost_summary": cost_summary,
        "table": quality_table,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_output, f, indent=2)
    print(f"Saved TASK R-03 report to {REPORT_PATH}", flush=True)

    return report_output

if __name__ == "__main__":
    evaluate_r03()
