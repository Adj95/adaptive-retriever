"""Experiment 003 — Query Transformation Evaluation (Task B: self_query).

Evaluates 169 non-rejected items on retriever_chunks_a5d51992 index across two global arms:
  Arm 1: strategy="none" (Baseline)
  Arm 2: strategy="self_query" (Attempt filter extraction on every query)

Outputs metrics, bootstrapped 95% CIs, and self_query filter extraction statistics to reports/experiment_003.json.
"""

import copy
import json
import math
import numpy as np
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import (
    bootstrap_ci,
    bootstrap_difference_ci,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
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


def run_experiment_003(smoke: bool = False):
    eval_items = load_eval_items()
    if smoke:
        eval_items = eval_items[:1]
    n_items = len(eval_items)
    print(f"Loaded {n_items} valid non-rejected eval items for Experiment 003 (smoke={smoke}).")

    base_cfg = Config.load(Path("config/default.yaml"))
    base_cfg.raw["funnel"]["rerank"]["gate_score_margin"] = 0.30
    base_cfg.raw["funnel"]["rerank"]["output_size"] = 10
    base_cfg.raw["funnel"]["hybrid"]["dense_limit"] = 100
    base_cfg.raw["funnel"]["hybrid"]["sparse_limit"] = 100

    vs = VectorStore.from_config(base_cfg)
    pts_count = vs.client.get_collection(vs.collection_name).points_count
    url_info = vs.url if vs.mode == "server" else str(base_cfg.raw.get("vectorstore", {}).get("storage_path", "data/qdrant_storage"))

    print("\n" + "=" * 80)
    print(" EXPERIMENT 003 HEADER")
    print(f"  Vectorstore Mode    : {vs.mode}")
    print(f"  Vectorstore URL/Path: {url_info}")
    print(f"  Collection Name     : {vs.collection_name}")
    print(f"  Collection Points   : {pts_count}")
    print("=" * 80 + "\n", flush=True)

    raw_reranker = Reranker(device=base_cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))
    raw_reranker.ensure_ready()
    reranker = CachedReranker(raw_reranker)
    warm_up(vs, reranker)

    # Define Arms
    cfg_none = Config.load(Path("config/default.yaml"))
    cfg_none.raw["query"]["transformation"]["enabled"] = True
    cfg_none.raw["query"]["transformation"]["fixed"] = "none"

    cfg_self = Config.load(Path("config/default.yaml"))
    cfg_self.raw["query"]["transformation"]["enabled"] = True
    cfg_self.raw["query"]["transformation"]["fixed"] = "self_query"

    arms = [
        ("Arm 1: none", cfg_none),
        ("Arm 2: self_query", cfg_self),
    ]

    arm_results: Dict[str, Dict[str, List[float]]] = {
        name: {
            "r5": [], "ndcg5": [], "r10": [], "ndcg10": [], "mrr": [],
            "ret_ms": [], "tot_ms": []
        }
        for name, _ in arms
    }

    # Clear diskcache to prevent stale cached transform results from contaminating the experiment
    cache_dir = base_cfg.raw.get("paths", {}).get("cache_dir", "data/cache")
    if Path(cache_dir).exists():
        from diskcache import Cache
        c = Cache(directory=cache_dir)
        c.clear()
        c.close()
        print(f"Cleared cache directory '{cache_dir}' for clean Experiment 003 run.")

    # Statistics specifically for self_query
    self_query_stats = {
        "filters_produced": 0,
        "filters_dropped_zero_docs": 0,
        "filters_rejected_unindexed": 0,
        "queries_zero_candidates": 0,
        "prefilter_output_counts": [],
    }

    filters_log_entries = []
    raw_log_records = []

    t0 = time.time()
    for idx, item in enumerate(eval_items, 1):
        query_text = item["query"]
        rel_passages_raw = item.get("relevant_passages", [])
        rel_passages = [RelevantPassage(**p) for p in rel_passages_raw]

        item_raw_record = {"item_idx": idx, "query": query_text, "arms": {}}

        for arm_name, cfg_arm in arms:
            cands, trace = run_funnel(query_text, vs=vs, reranker=reranker, cfg=cfg_arm)
            pred_texts = [c.text for c in cands]

            r5 = recall_at_k(pred_texts, rel_passages, k=5)
            n5 = ndcg_at_k(pred_texts, rel_passages, k=5)
            r10 = recall_at_k(pred_texts, rel_passages, k=10)
            n10 = ndcg_at_k(pred_texts, rel_passages, k=10)
            m_rr = mrr(pred_texts, rel_passages)

            arm_results[arm_name]["r5"].append(r5)
            arm_results[arm_name]["ndcg5"].append(n5)
            arm_results[arm_name]["r10"].append(r10)
            arm_results[arm_name]["ndcg10"].append(n10)
            arm_results[arm_name]["mrr"].append(m_rr)

            stages = trace.get("stages", {})
            ret_ms = (
                stages.get("dense_search", {}).get("elapsed_ms", 0.0)
                + stages.get("sparse_search", {}).get("elapsed_ms", 0.0)
                + stages.get("fusion", {}).get("elapsed_ms", 0.0)
            )
            tot_ms = trace.get("total_elapsed_ms", 0.0)
            arm_results[arm_name]["ret_ms"].append(ret_ms)
            arm_results[arm_name]["tot_ms"].append(tot_ms)

            item_raw_record["arms"][arm_name] = {
                "r5": r5, "ndcg5": n5, "r10": r10, "ndcg10": n10, "mrr": m_rr,
                "ret_ms": ret_ms, "tot_ms": tot_ms,
                "candidate_count": len(cands),
                "prefilter_count": stages.get("prefilter", {}).get("output_count", 0),
            }

            if arm_name == "Arm 2: self_query":
                trans_stage = stages.get("transformation", {})
                det = trans_stage.get("details", {})
                if isinstance(det, dict) and "details" in det and isinstance(det["details"], dict):
                    inner_det = det["details"]
                else:
                    inner_det = det if isinstance(det, dict) else {}

                cand_filters = trans_stage.get("candidate_filters") or inner_det.get("candidate_filters") or {}
                dropped_reason = trans_stage.get("dropped_reason") or inner_det.get("dropped_reason")
                rejected = trans_stage.get("rejected_fields") or inner_det.get("rejected_fields") or {}
                match_cnt = trans_stage.get("match_count") if trans_stage.get("match_count") is not None else inner_det.get("match_count")

                pref_stage = stages.get("prefilter", {})
                pf_output_cnt = pref_stage.get("output_count", 0)
                final_cand_cnt = len(cands)

                if cand_filters:
                    self_query_stats["filters_produced"] += 1
                if dropped_reason == "matches_no_documents":
                    self_query_stats["filters_dropped_zero_docs"] += 1
                if rejected:
                    self_query_stats["filters_rejected_unindexed"] += 1
                if final_cand_cnt == 0:
                    self_query_stats["queries_zero_candidates"] += 1

                if pf_output_cnt is not None:
                    self_query_stats["prefilter_output_counts"].append(pf_output_cnt)

                log_entry = {
                    "query": query_text,
                    "extracted_filters": cand_filters,
                    "match_count": match_cnt,
                    "guard_dropped": dropped_reason == "matches_no_documents",
                    "final_prefilter_output_count": pf_output_cnt,
                    "final_candidate_count": final_cand_cnt,
                }
                filters_log_entries.append(log_entry)

        raw_log_records.append(item_raw_record)

        if idx % 20 == 0 or idx == n_items:
            print(f"  Processed {idx}/{n_items} queries... ({time.time() - t0:.1f}s)", flush=True)

    # Write per-query audit log & raw records
    filters_log_path = Path("reports/experiment_003_filters.jsonl")
    filters_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filters_log_path, "w", encoding="utf-8") as f:
        for entry in filters_log_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved filter audit log to {filters_log_path}")

    raw_log_path = Path("reports/experiment_003_raw.jsonl")
    with open(raw_log_path, "w", encoding="utf-8") as f:
        for rec in raw_log_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved raw per-query run records to {raw_log_path}")

    # Compute metric summaries
    seed = int(base_cfg.raw.get("seed", 42))
    report = {
        "metadata": {
            "n_items": n_items,
            "vectorstore_mode": vs.mode,
            "collection_name": vs.collection_name,
            "collection_points": pts_count,
            "seed": seed,
        },
        "arms": {},
        "differences": {},
        "self_query_stats": {},
    }

    metrics_keys = ["r5", "ndcg5", "r10", "ndcg10", "mrr"]
    for arm_name, _ in arms:
        arm_dict = {}
        for mk in metrics_keys:
            vals = arm_results[arm_name][mk]
            mean_val, ci_low, ci_high = bootstrap_ci(vals, n=1000, ci=0.95, seed=seed)
            arm_dict[mk] = {
                "mean": round(float(mean_val), 4),
                "ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
            }
        arm_dict["retrieval_lat_ms_p50"] = round(float(np.percentile(arm_results[arm_name]["ret_ms"], 50)), 2)
        arm_dict["total_lat_ms_p50"] = round(float(np.percentile(arm_results[arm_name]["tot_ms"], 50)), 2)
        report["arms"][arm_name] = arm_dict

    # Differences: Arm 2 (self_query) - Arm 1 (none)
    diff_dict = {}
    for mk in metrics_keys:
        v1 = arm_results["Arm 1: none"][mk]
        v2 = arm_results["Arm 2: self_query"][mk]
        diff_mean, d_low, d_high = bootstrap_difference_ci(v2, v1, n=1000, ci=0.95, seed=seed)
        diff_dict[mk] = {
            "mean_diff": round(float(diff_mean), 4),
            "ci_95": [round(float(d_low), 4), round(float(d_high), 4)],
        }
    report["differences"]["self_query_vs_none"] = diff_dict

    # Prefilter counts stats
    pref_counts = self_query_stats["prefilter_output_counts"]
    if pref_counts:
        report["self_query_stats"] = {
            "total_queries": n_items,
            "filters_produced": self_query_stats["filters_produced"],
            "filters_dropped_zero_docs": self_query_stats["filters_dropped_zero_docs"],
            "filters_rejected_unindexed": self_query_stats["filters_rejected_unindexed"],
            "queries_zero_candidates": self_query_stats["queries_zero_candidates"],
            "prefilter_count_distribution": {
                "min": int(np.min(pref_counts)),
                "p25": round(float(np.percentile(pref_counts, 25)), 1),
                "median": round(float(np.median(pref_counts)), 1),
                "p75": round(float(np.percentile(pref_counts, 75)), 1),
                "max": int(np.max(pref_counts)),
                "mean": round(float(np.mean(pref_counts)), 1),
            },
        }

    # MANDATORY MECHANICAL CONSISTENCY ASSERTIONS (standing rule)
    assert self_query_stats["filters_dropped_zero_docs"] <= self_query_stats["filters_produced"], (
        f"Contradictory stats: filters_dropped ({self_query_stats['filters_dropped_zero_docs']}) > "
        f"filters_produced ({self_query_stats['filters_produced']})"
    )
    if self_query_stats["filters_produced"] == 0:
        assert self_query_stats["filters_dropped_zero_docs"] == 0, (
            f"Contradictory stats: filters_produced is 0 but filters_dropped is {self_query_stats['filters_dropped_zero_docs']}"
        )

    if self_query_stats["filters_produced"] == 0 and self_query_stats["queries_zero_candidates"] == 0:
        dist = report["self_query_stats"]["prefilter_count_distribution"]
        assert abs(dist["mean"] - pts_count) < 1e-4, f"Prefilter mean is {dist['mean']}, expected {pts_count}"
        assert abs(dist["median"] - pts_count) < 1e-4, f"Prefilter median is {dist['median']}, expected {pts_count}"
        for mk in metrics_keys:
            diff_val = report["differences"]["self_query_vs_none"][mk]["mean_diff"]
            assert abs(diff_val) == 0.0, (
                f"Contradictory report: self_query produced 0 filters, yet difference for {mk} is {diff_val}, expected 0.0000"
            )

    # Print Summary Table
    print("\n" + "=" * 80)
    print(" EXPERIMENT 003 SUMMARY: TASK B (self_query vs none)")
    print("=" * 80)
    print(f"{'Metric':<12} | {'Arm 1: none':<18} | {'Arm 2: self_query':<18} | {'Difference [95% CI]':<26}")
    print("-" * 80)
    for mk, name in [("r5", "Recall@5"), ("ndcg5", "nDCG@5"), ("r10", "Recall@10"), ("ndcg10", "nDCG@10"), ("mrr", "MRR")]:
        a1 = report["arms"]["Arm 1: none"][mk]
        a2 = report["arms"]["Arm 2: self_query"][mk]
        df = report["differences"]["self_query_vs_none"][mk]
        a1_str = f"{a1['mean']:.4f} [{a1['ci_95'][0]:.4f}, {a1['ci_95'][1]:.4f}]"
        a2_str = f"{a2['mean']:.4f} [{a2['ci_95'][0]:.4f}, {a2['ci_95'][1]:.4f}]"
        df_str = f"{df['mean_diff']:+.4f} [{df['ci_95'][0]:+.4f}, {df['ci_95'][1]:+.4f}]"
        print(f"{name:<12} | {a1_str:<18} | {a2_str:<18} | {df_str:<26}")
    print("-" * 80)
    print(f"Self-Query Statistics:")
    print(f"  Filters produced           : {report['self_query_stats'].get('filters_produced', 0)} / {n_items}")
    print(f"  Dropped (0 matching docs)  : {report['self_query_stats'].get('filters_dropped_zero_docs', 0)}")
    print(f"  Rejected unindexed fields  : {report['self_query_stats'].get('filters_rejected_unindexed', 0)}")
    print(f"  Queries with 0 candidates  : {report['self_query_stats'].get('queries_zero_candidates', 0)}")
    print(f"  Prefilter count median/mean: {report['self_query_stats'].get('prefilter_count_distribution', {}).get('median')} / {report['self_query_stats'].get('prefilter_count_distribution', {}).get('mean')}")
    print("=" * 80 + "\n")

    # Save to file
    out_path = Path("reports/experiment_003.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Experiment 003 results to {out_path}")


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    run_experiment_003(smoke=smoke)
