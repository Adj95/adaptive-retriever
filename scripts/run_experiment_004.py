"""Experiment 004 — Do per-class fusion weights matter?

Evaluates 169 non-rejected items on retriever_chunks_a5d51992 index across 4 arms:
  Arm 1: current per-class routing (profiler picks weights) — Phase 2 shipped arm
  Arm 2: fixed {dense: 0.5, sparse: 0.5} for every query
  Arm 3: fixed {dense: 0.8, sparse: 0.2} for every query
  Arm 4: fixed {dense: 0.3, sparse: 0.7} for every query

Outputs metrics and paired bootstrap CIs to reports/experiment_004.json.
"""

import copy
import json
import math
import numpy as np
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import bootstrap_ci, bootstrap_difference_ci, mrr, ndcg_at_k, recall_at_k
from adaptive_retriever.funnel.pipeline import run_funnel, warm_up
from adaptive_retriever.funnel.rerank import Reranker
from adaptive_retriever.models import RelevantPassage
from adaptive_retriever.vectorstore import VectorStore


def load_eval_items(eval_path: Path = Path("data/eval/eval_set.jsonl")) -> List[Dict[str, Any]]:
    items = []
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("provenance") != "synthetic_rejected":
            items.append(item)
    return items


class CachedReranker:
    def __init__(self, inner: Reranker):
        self.inner = inner
        self.cache: Dict[Tuple[str, str], float] = {}

    def ensure_ready(self) -> None:
        self.inner.ensure_ready()

    def score(self, query: str, texts: List[str]) -> List[float]:
        missing = [t for t in texts if (query, t) not in self.cache]
        if missing:
            new_scores = self.inner.score(query, missing)
            for t, s in zip(missing, new_scores):
                self.cache[(query, t)] = s
        return [self.cache[(query, t)] for t in texts]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def run_experiment_004():
    eval_items = load_eval_items()
    n_items = len(eval_items)
    print(f"Loaded {n_items} valid non-rejected eval items for Experiment 004.")

    base_cfg = Config.load(Path("config/default.yaml"))
    base_cfg.raw["funnel"]["rerank"]["gate_score_margin"] = 0.30
    base_cfg.raw["funnel"]["rerank"]["output_size"] = 10
    base_cfg.raw["funnel"]["hybrid"]["dense_limit"] = 100
    base_cfg.raw["funnel"]["hybrid"]["sparse_limit"] = 100

    vs = VectorStore.from_config(base_cfg)
    pts_count = vs.client.get_collection(vs.collection_name).points_count
    url_info = vs.url if vs.mode == "server" else str(base_cfg.raw.get("vectorstore", {}).get("storage_path", "data/qdrant_storage"))

    print("\n" + "=" * 80)
    print(" EXPERIMENT 004 HEADER")
    print(f"  Vectorstore Mode    : {vs.mode}")
    print(f"  Vectorstore URL/Path: {url_info}")
    print(f"  Collection Name     : {vs.collection_name}")
    print(f"  Collection Points   : {pts_count}")
    print("=" * 80 + "\n", flush=True)

    raw_reranker = Reranker(device=base_cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))
    raw_reranker.ensure_ready()
    reranker = CachedReranker(raw_reranker)
    warm_up(vs, reranker)

    all_classes = ["exact_lookup", "conceptual", "compound", "multihop", "aggregation", "filterable", "default"]

    arm1_weights = copy.deepcopy(base_cfg.raw.get("funnel", {}).get("hybrid", {}).get("class_weights", {}))
    arm2_weights = {c: {"dense": 0.5, "sparse": 0.5} for c in all_classes}
    arm3_weights = {c: {"dense": 0.8, "sparse": 0.2} for c in all_classes}
    arm4_weights = {c: {"dense": 0.3, "sparse": 0.7} for c in all_classes}

    arms = [
        ("Arm 1: Per-Class Routing", arm1_weights),
        ("Arm 2: Fixed 0.5/0.5", arm2_weights),
        ("Arm 3: Fixed 0.8/0.2", arm3_weights),
        ("Arm 4: Fixed 0.3/0.7", arm4_weights),
    ]

    arm_configs = []
    for arm_name, weights_dict in arms:
        cfg_arm = Config.load(Path("config/default.yaml"))
        cfg_arm.raw["funnel"]["rerank"]["gate_score_margin"] = 0.30
        cfg_arm.raw["funnel"]["rerank"]["output_size"] = 10
        cfg_arm.raw["funnel"]["hybrid"]["class_weights"] = weights_dict
        arm_configs.append((arm_name, cfg_arm))

    arm_results: Dict[str, Dict[str, List[float]]] = {
        arm_name: {"recall": [], "ndcg": [], "mrr": []} for arm_name, _ in arms
    }

    t0 = time.time()
    for idx, item in enumerate(eval_items, 1):
        query_text = item["query"]
        rel_passages_raw = item.get("relevant_passages", [])
        rel_passages = [RelevantPassage(**p) for p in rel_passages_raw]

        for arm_name, cfg_arm in arm_configs:
            candidates, _trace = run_funnel(
                query_text=query_text,
                vs=vs,
                cfg=cfg_arm,
                reranker=reranker,
            )

            cand_texts = [cand.text for cand in candidates]

            rec_10 = recall_at_k(cand_texts, rel_passages, k=10)
            ndcg_10 = ndcg_at_k(cand_texts, rel_passages, k=10)
            mrr_val = mrr(cand_texts, rel_passages)

            arm_results[arm_name]["recall"].append(rec_10)
            arm_results[arm_name]["ndcg"].append(ndcg_10)
            arm_results[arm_name]["mrr"].append(mrr_val)

        if idx % 10 == 0 or idx == n_items:
            elapsed = time.time() - t0
            print(f"Evaluated {idx}/{n_items} items ({elapsed:.1f}s)...", flush=True)

    # Paired Bootstrap Analysis (1,000 resamples, seed=42)
    summary_metrics: Dict[str, Dict[str, Any]] = {}
    for arm_name in arm_results:
        rec_val, rec_lo, rec_hi = bootstrap_ci(arm_results[arm_name]["recall"], n=1000, seed=42)
        ndcg_val, ndcg_lo, ndcg_hi = bootstrap_ci(arm_results[arm_name]["ndcg"], n=1000, seed=42)
        mrr_val, mrr_lo, mrr_hi = bootstrap_ci(arm_results[arm_name]["mrr"], n=1000, seed=42)

        summary_metrics[arm_name] = {
            "recall_10": {
                "mean": round(rec_val, 4),
                "ci95": [round(rec_lo, 4), round(rec_hi, 4)],
            },
            "ndcg_10": {
                "mean": round(ndcg_val, 4),
                "ci95": [round(ndcg_lo, 4), round(ndcg_hi, 4)],
            },
            "mrr": {
                "mean": round(mrr_val, 4),
                "ci95": [round(mrr_lo, 4), round(mrr_hi, 4)],
            },
        }

    # Paired differences against Arm 1
    arm1_rec = arm_results["Arm 1: Per-Class Routing"]["recall"]
    arm1_ndcg = arm_results["Arm 1: Per-Class Routing"]["ndcg"]
    arm1_mrr = arm_results["Arm 1: Per-Class Routing"]["mrr"]

    paired_diffs: Dict[str, Dict[str, Any]] = {}
    compared_arms = [
        ("Arm 2: Fixed 0.5/0.5", "Arm 2 (0.5/0.5) vs Arm 1"),
        ("Arm 3: Fixed 0.8/0.2", "Arm 3 (0.8/0.2) vs Arm 1"),
        ("Arm 4: Fixed 0.3/0.7", "Arm 4 (0.3/0.7) vs Arm 1"),
    ]

    total_comparisons = len(compared_arms) * 3  # 3 arms x 3 metrics = 9
    alpha_bonferroni = 0.05 / total_comparisons

    # Deterministic paired resample for p-values
    rng = np.random.default_rng(42)
    n_resamples = 1000
    boot_indices = [rng.choice(n_items, size=n_items, replace=True) for _ in range(n_resamples)]

    for arm_key, comp_label in compared_arms:
        arm_rec = arm_results[arm_key]["recall"]
        arm_ndcg = arm_results[arm_key]["ndcg"]
        arm_mrr = arm_results[arm_key]["mrr"]

        rec_diff_val, rec_diff_lo, rec_diff_hi = bootstrap_difference_ci(arm_rec, arm1_rec, n=1000, seed=42)
        ndcg_diff_val, ndcg_diff_lo, ndcg_diff_hi = bootstrap_difference_ci(arm_ndcg, arm1_ndcg, n=1000, seed=42)
        mrr_diff_val, mrr_diff_lo, mrr_diff_hi = bootstrap_difference_ci(arm_mrr, arm1_mrr, n=1000, seed=42)

        # Empirical p-values across paired bootstrap samples
        d_rec = np.array(arm_rec) - np.array(arm1_rec)
        d_ndcg = np.array(arm_ndcg) - np.array(arm1_ndcg)
        d_mrr = np.array(arm_mrr) - np.array(arm1_mrr)

        rec_bs_diffs = [np.mean(d_rec[idx]) for idx in boot_indices]
        ndcg_bs_diffs = [np.mean(d_ndcg[idx]) for idx in boot_indices]
        mrr_bs_diffs = [np.mean(d_mrr[idx]) for idx in boot_indices]

        p_rec = float(2 * min(np.mean(np.array(rec_bs_diffs) <= 0), np.mean(np.array(rec_bs_diffs) >= 0)))
        p_ndcg = float(2 * min(np.mean(np.array(ndcg_bs_diffs) <= 0), np.mean(np.array(ndcg_bs_diffs) >= 0)))
        p_mrr = float(2 * min(np.mean(np.array(mrr_bs_diffs) <= 0), np.mean(np.array(mrr_bs_diffs) >= 0)))

        paired_diffs[comp_label] = {
            "recall_10_diff": {
                "mean_diff": round(rec_diff_val, 4),
                "ci95": [round(rec_diff_lo, 4), round(rec_diff_hi, 4)],
                "p_value": round(p_rec, 4),
                "sig_bonferroni": p_rec < alpha_bonferroni,
            },
            "ndcg_10_diff": {
                "mean_diff": round(ndcg_diff_val, 4),
                "ci95": [round(ndcg_diff_lo, 4), round(ndcg_diff_hi, 4)],
                "p_value": round(p_ndcg, 4),
                "sig_bonferroni": p_ndcg < alpha_bonferroni,
            },
            "mrr_diff": {
                "mean_diff": round(mrr_diff_val, 4),
                "ci95": [round(mrr_diff_lo, 4), round(mrr_diff_hi, 4)],
                "p_value": round(p_mrr, 4),
                "sig_bonferroni": p_mrr < alpha_bonferroni,
            },
        }

    # Save to reports/experiment_004.json
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "experiment_004.json"

    report_payload = {
        "meta": {
            "n_items": n_items,
            "gate_score_margin": 0.30,
            "output_size": 10,
            "total_comparisons": total_comparisons,
            "bonferroni_alpha": round(alpha_bonferroni, 5),
            "multiplicity_note": f"Evaluated {total_comparisons} paired comparisons (3 arms x 3 metrics) at alpha={alpha_bonferroni:.5f}.",
        },
        "summary_metrics": summary_metrics,
        "paired_differences": paired_diffs,
    }

    report_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nSaved Experiment 004 report to {report_file}")

    # Print nicely formatted summary table
    print("\n" + "=" * 115)
    print(" EXPERIMENT 004 — DO PER-CLASS FUSION WEIGHTS MATTER?")
    print(" (n = 169 non-rejected items | gate_score_margin: 0.30 | retrieval depth: 10)")
    print("=" * 115)
    print(f"{'Arm Name':<32} | {'Recall@10 [95% CI]':<26} | {'nDCG@10 [95% CI]':<26} | {'MRR [95% CI]':<26}")
    print("-" * 115)

    for arm_name in summary_metrics:
        m = summary_metrics[arm_name]
        r_str = f"{m['recall_10']['mean']:.4f} [{m['recall_10']['ci95'][0]:.4f}, {m['recall_10']['ci95'][1]:.4f}]"
        n_str = f"{m['ndcg_10']['mean']:.4f} [{m['ndcg_10']['ci95'][0]:.4f}, {m['ndcg_10']['ci95'][1]:.4f}]"
        m_str = f"{m['mrr']['mean']:.4f} [{m['mrr']['ci95'][0]:.4f}, {m['mrr']['ci95'][1]:.4f}]"
        print(f"{arm_name:<32} | {r_str:<26} | {n_str:<26} | {m_str:<26}")

    print("=" * 115)
    print("\nPAIRED BOOTSTRAP DIFFERENCES AGAINST ARM 1 (PER-CLASS ROUTING):")
    print("-" * 115)
    print(f"{'Comparison':<28} | {'Metric':<10} | {'Mean Diff':<10} | {'95% CI':<20} | {'p-value':<10} | {'Sig (alpha=0.0056)'}")
    print("-" * 115)

    for comp_label, diffs in paired_diffs.items():
        for metric_key, metric_title in [("recall_10_diff", "Recall@10"), ("ndcg_10_diff", "nDCG@10"), ("mrr_diff", "MRR")]:
            d = diffs[metric_key]
            ci_str = f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
            sig_str = "YES" if d["sig_bonferroni"] else "NO"
            print(f"{comp_label:<28} | {metric_title:<10} | {d['mean_diff']:+.4f}    | {ci_str:<20} | {d['p_value']:<10.4f} | {sig_str}")


if __name__ == "__main__":
    run_experiment_004()
