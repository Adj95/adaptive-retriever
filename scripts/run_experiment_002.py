"""Run Experiment 002: Rerank Gate Margin Sweep.

Sweeps gate_score_margin over [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 1.00] over 169 evaluation items.
For each value reports:
- gate fire rate
- Recall@10, nDCG@10, MRR (with 95% CIs)
- Paired bootstrap differences against margin = 0.00 (with 95% CIs)
- Retrieval, rerank, and total latencies (mean, p50, and p95)
Outputs results to reports/experiment_002.json and prints formatted stdout table.
"""

import copy
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
from adaptive_retriever.funnel.pipeline import warm_up
from adaptive_retriever.funnel.prefilter import build_prefilter
from adaptive_retriever.funnel.rerank import Reranker, apply_rerank, should_rerank
from adaptive_retriever.models import EvalItem, RetrievalCandidate
from adaptive_retriever.query.profiler import profile_query
from adaptive_retriever.vectorstore import VectorStore


def load_eval_set(eval_path: Path) -> List[EvalItem]:
    with open(eval_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [EvalItem(**rec) for rec in records]


def main():
    t_exp_start = time.perf_counter()
    base_cfg = Config.load(Path("config/default.yaml"))
    seed = int(base_cfg.raw.get("seed", 42))
    bootstrap_samples = int(base_cfg.raw.get("eval", {}).get("bootstrap_samples", 1000))
    ci_level = float(base_cfg.raw.get("eval", {}).get("ci", 0.95))

    eval_path = Path(base_cfg.raw.get("paths", {}).get("eval_set", "data/eval/eval_set.jsonl"))
    all_items = load_eval_set(eval_path)

    # Filter out synthetic_rejected
    valid_items = [item for item in all_items if item.provenance != "synthetic_rejected"]
    excluded_items = [item for item in all_items if item.provenance == "synthetic_rejected"]

    # Shared vectorstore and reranker in ONE warmed process
    vs = VectorStore.from_config(base_cfg)
    reranker = Reranker(device=base_cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))

    print("=== WARMING UP MODELS & QDRANT CLIENT CONNECTION ===")
    warmup_ms = warm_up(vs, reranker)
    print(f"Warm-up complete in {warmup_ms:.2f} ms | Reranker device: {reranker.get_device_str()}")

    margins = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 1.00]
    print(f"\nExecuting Gate Margin Sweep over {len(valid_items)} items for margins: {margins}...")

    # Data collection per margin
    sweep_data: Dict[float, Dict[str, List]] = {
        m: {
            "r10": [],
            "ndcg10": [],
            "mrr": [],
            "ret_ms": [],
            "rr_ms": [],
            "tot_ms": [],
            "fired": [],
        }
        for m in margins
    }

    output_size = int(base_cfg.raw.get("funnel", {}).get("rerank", {}).get("output_size", 10))
    hybrid_limit = int(base_cfg.raw.get("funnel", {}).get("hybrid", {}).get("dense_limit", 100))

    for idx, item in enumerate(valid_items, 1):
        query_text = item.query
        profile = profile_query(query_text)
        query_cls = profile.query_class.value if hasattr(profile.query_class, "value") else str(profile.query_class)

        # Step 1: Pre-filter + Hybrid Search (common across all margins)
        t0_ret = time.perf_counter()
        filter_obj, _ = build_prefilter(profile, cfg=base_cfg)
        hybrid_cands = vs.query_hybrid(
            query_text=query_text,
            limit=hybrid_limit,
            query_class=query_cls,
            cfg=base_cfg,
            query_filter=filter_obj,
        )
        t1_ret = time.perf_counter()
        ret_ms = (t1_ret - t0_ret) * 1000.0

        # Cache cross-encoder reranked candidates if needed for this query
        cached_reranked_cands = None
        cached_rr_ms = 0.0

        # Step 2: Sweep each margin setting
        for m in margins:
            cfg_m = copy.deepcopy(base_cfg.raw)
            cfg_m["funnel"]["rerank"]["gate_score_margin"] = m
            if m == 0.00:
                cfg_m["funnel"]["rerank"]["mode"] = "never"
            elif m == 1.00:
                cfg_m["funnel"]["rerank"]["mode"] = "always"
            else:
                cfg_m["funnel"]["rerank"]["mode"] = "gated"

            fire, _ = should_rerank(hybrid_cands, cfg=cfg_m)

            if fire:
                if cached_reranked_cands is None:
                    t0_rr = time.perf_counter()
                    cached_reranked_cands = apply_rerank(query_text, hybrid_cands, reranker, cfg=cfg_m)
                    t1_rr = time.perf_counter()
                    cached_rr_ms = (t1_rr - t0_rr) * 1000.0
                cands = cached_reranked_cands[:output_size]
                rr_ms = cached_rr_ms
            else:
                top_cands = hybrid_cands[:output_size]
                cands = [c.model_copy(update={"final_rank": r}) for r, c in enumerate(top_cands, 1)]
                rr_ms = 0.0

            chunk_texts = [c.text for c in cands]
            sweep_data[m]["r10"].append(recall_at_k(chunk_texts, item.relevant_passages, k=10))
            sweep_data[m]["ndcg10"].append(ndcg_at_k(chunk_texts, item.relevant_passages, k=10))
            sweep_data[m]["mrr"].append(mrr(chunk_texts, item.relevant_passages))
            sweep_data[m]["ret_ms"].append(ret_ms)
            sweep_data[m]["rr_ms"].append(rr_ms)
            sweep_data[m]["tot_ms"].append(ret_ms + rr_ms)
            sweep_data[m]["fired"].append(1 if fire else 0)

        if idx % 25 == 0 or idx == len(valid_items):
            print(f"  Processed {idx}/{len(valid_items)} items...")

    # Baseline lists for paired bootstrap differences against margin = 0.00
    base_r10_list = sweep_data[0.00]["r10"]
    base_ndcg10_list = sweep_data[0.00]["ndcg10"]
    base_mrr_list = sweep_data[0.00]["mrr"]

    # Process sweep results, CIs, paired differences, and mean latencies
    margin_results = {}

    for m in margins:
        data = sweep_data[m]
        fired_count = sum(data["fired"])
        fire_rate = float(fired_count / len(valid_items))

        ci_r10 = bootstrap_ci(data["r10"], n=bootstrap_samples, ci=ci_level, seed=seed)
        ci_ndcg10 = bootstrap_ci(data["ndcg10"], n=bootstrap_samples, ci=ci_level, seed=seed)
        ci_mrr = bootstrap_ci(data["mrr"], n=bootstrap_samples, ci=ci_level, seed=seed)

        diff_r10 = bootstrap_difference_ci(data["r10"], base_r10_list, n=bootstrap_samples, ci=ci_level, seed=seed)
        diff_ndcg10 = bootstrap_difference_ci(data["ndcg10"], base_ndcg10_list, n=bootstrap_samples, ci=ci_level, seed=seed)
        diff_mrr = bootstrap_difference_ci(data["mrr"], base_mrr_list, n=bootstrap_samples, ci=ci_level, seed=seed)

        ret_mean = float(np.mean(data["ret_ms"]))
        ret_p50 = float(np.percentile(data["ret_ms"], 50))
        ret_p95 = float(np.percentile(data["ret_ms"], 95))

        rr_mean = float(np.mean(data["rr_ms"]))
        rr_p50 = float(np.percentile(data["rr_ms"], 50))
        rr_p95 = float(np.percentile(data["rr_ms"], 95))

        tot_mean = float(np.mean(data["tot_ms"]))
        tot_p50 = float(np.percentile(data["tot_ms"], 50))
        tot_p95 = float(np.percentile(data["tot_ms"], 95))

        # Check if paired CI excludes zero for nDCG@10 or MRR
        ndcg_sig = (diff_ndcg10[1] > 0.0) or (diff_ndcg10[2] < 0.0)
        mrr_sig = (diff_mrr[1] > 0.0) or (diff_mrr[2] < 0.0)
        r10_sig = (diff_r10[1] > 0.0) or (diff_r10[2] < 0.0)

        res_entry = {
            "gate_score_margin": m,
            "fire_rate": fire_rate,
            "fired_count": fired_count,
            "recall_10": {"mean": ci_r10[0], "ci_95": [ci_r10[1], ci_r10[2]]},
            "ndcg_10": {"mean": ci_ndcg10[0], "ci_95": [ci_ndcg10[1], ci_ndcg10[2]]},
            "mrr": {"mean": ci_mrr[0], "ci_95": [ci_mrr[1], ci_mrr[2]]},
            "paired_diff_vs_000": {
                "recall_10_diff": {"mean": diff_r10[0], "ci_95": [diff_r10[1], diff_r10[2]], "statistically_significant": r10_sig},
                "ndcg_10_diff": {"mean": diff_ndcg10[0], "ci_95": [diff_ndcg10[1], diff_ndcg10[2]], "statistically_significant": ndcg_sig},
                "mrr_diff": {"mean": diff_mrr[0], "ci_95": [diff_mrr[1], diff_mrr[2]], "statistically_significant": mrr_sig},
            },
            "retrieval_latency_ms": {"mean": ret_mean, "p50": ret_p50, "p95": ret_p95},
            "rerank_latency_ms": {"mean": rr_mean, "p50": rr_p50, "p95": rr_p95},
            "total_latency_ms": {"mean": tot_mean, "p50": tot_p50, "p95": tot_p95},
        }
        margin_results[str(m)] = res_entry

    t_exp_end = time.perf_counter()
    exp_elapsed_s = round(t_exp_end - t_exp_start, 2)

    experiment_report = {
        "meta": {
            "experiment_id": "experiment_002",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "ci_level": ci_level,
            "items_evaluated": len(valid_items),
            "items_excluded": len(excluded_items),
            "total_experiment_time_seconds": exp_elapsed_s,
        },
        "sweep": margin_results,
    }

    # Save to reports/experiment_002.json
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "experiment_002.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)

    print(f"\nSaved experiment 002 results to {out_file}\n")

    # Format table output for stdout
    fmt_ci = lambda mean, low, high: f"{mean:.4f} [{low:.4f}, {high:.4f}]"
    fmt_diff = lambda mean, low, high: f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"

    print("=" * 165)
    print(" EXPERIMENT 002 RESULTS: RERANK GATE MARGIN SWEEP (WITH PAIRED DIFFERENCES VS MARGIN 0.00 & MEAN LATENCIES)")
    print(f" (Seed: {seed} | Resample Iterations: {bootstrap_samples} | Evaluated Items: {len(valid_items)})")
    print("=" * 165)
    print(
        f"{'Margin':<6} | {'FireRate':<8} | {'Recall@10 (Paired Diff vs 0.00)':<34} | "
        f"{'nDCG@10 (Paired Diff vs 0.00)':<34} | {'MRR (Paired Diff vs 0.00)':<34} | "
        f"{'RR Lat (mean/p50/p95)':<22} | {'Tot Lat (mean/p50/p95)':<22}"
    )
    print("-" * 165)

    for m in margins:
        entry = margin_results[str(m)]
        fr_str = f"{entry['fire_rate']*100:.1f}%"
        
        diff_r = entry["paired_diff_vs_000"]["recall_10_diff"]
        diff_n = entry["paired_diff_vs_000"]["ndcg_10_diff"]
        diff_m = entry["paired_diff_vs_000"]["mrr_diff"]

        r10_str = f"{entry['recall_10']['mean']:.4f} ({fmt_diff(diff_r['mean'], diff_r['ci_95'][0], diff_r['ci_95'][1])})"
        ndcg10_str = f"{entry['ndcg_10']['mean']:.4f} ({fmt_diff(diff_n['mean'], diff_n['ci_95'][0], diff_n['ci_95'][1])})"
        mrr_str = f"{entry['mrr']['mean']:.4f} ({fmt_diff(diff_m['mean'], diff_m['ci_95'][0], diff_m['ci_95'][1])})"

        rr_lat_str = f"{entry['rerank_latency_ms']['mean']:.1f}/{entry['rerank_latency_ms']['p50']:.1f}/{entry['rerank_latency_ms']['p95']:.1f}"
        tot_lat_str = f"{entry['total_latency_ms']['mean']:.1f}/{entry['total_latency_ms']['p50']:.1f}/{entry['total_latency_ms']['p95']:.1f}"

        print(f"{m:<6.2f} | {fr_str:<8} | {r10_str:<34} | {ndcg10_str:<34} | {mrr_str:<34} | {rr_lat_str:<22} | {tot_lat_str:<22}")

    print("=" * 165 + "\n")


if __name__ == "__main__":
    main()
