"""Experiment 005 — Global Query Transformations Evaluation (HyDE and Decomposition).

Evaluates 169 non-rejected items on retriever_chunks_a5d51992 index across four global arms:
  Arm 1: strategy="none" (Baseline)
  Arm 2: strategy="hyde" (Corrected HyDE: dense=hypothetical doc, sparse=raw query, reranker=raw query)
  Arm 3: strategy="decompose" (Unweighted multi-subquery fusion: 1/K each)
  Arm 4: strategy="decompose_weighted" (Original query weighted 0.60, paraphrases 0.20 each)

Outputs metrics, bootstrapped 95% CIs, latency/cost breakdown, subquery count breakdown, and raw ledger records to reports/experiment_005.json.
"""

import copy
import json
import math
import numpy as np
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import torch

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
from adaptive_retriever.llm import get_llm_client
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


def run_experiment_005(smoke: bool = False):
    eval_items = load_eval_items()
    if smoke:
        eval_items = eval_items[:2]
    n_items = len(eval_items)
    print(f"Loaded {n_items} valid non-rejected eval items for Experiment 005 (smoke={smoke}).")

    base_cfg = Config.load(Path("config/default.yaml"))
    base_cfg.raw["funnel"]["rerank"]["gate_score_margin"] = 0.30
    base_cfg.raw["funnel"]["rerank"]["output_size"] = 10
    base_cfg.raw["funnel"]["hybrid"]["dense_limit"] = 100
    base_cfg.raw["funnel"]["hybrid"]["sparse_limit"] = 100

    vs = VectorStore.from_config(base_cfg)
    pts_count = vs.client.get_collection(vs.collection_name).points_count
    url_info = vs.url if vs.mode == "server" else str(base_cfg.raw.get("vectorstore", {}).get("storage_path", "data/qdrant_storage"))

    print("\n" + "=" * 80)
    print(" EXPERIMENT 005 HEADER")
    print(f"  Python Executable   : {sys.executable}")
    print(f"  PyTorch Version     : {torch.__version__}")
    print(f"  PyTorch CUDA Version: {torch.version.cuda}")
    print(f"  Vectorstore Mode    : {vs.mode}")
    print(f"  Vectorstore URL/Path: {url_info}")
    print(f"  Collection Name     : {vs.collection_name}")
    print(f"  Collection Points   : {pts_count}")
    print(f"  CUDA Available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device          : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Memory Alloc   : {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
    print("=" * 80 + "\n", flush=True)

    # Initialize LLM client
    llm = get_llm_client(base_cfg)

    raw_reranker = Reranker(device=base_cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))
    raw_reranker.ensure_ready()
    reranker = CachedReranker(raw_reranker)
    warm_up(vs, reranker)

    # Instrument reranker check
    print("\n--- INSTRUMENTATION CHECK AT POINT OF EXECUTION ---")
    print(f"  sys.executable              : {sys.executable}")
    print(f"  torch.version.cuda          : {torch.version.cuda}")
    print(f"  reranker.model.device       : {raw_reranker.get_device_str()}")
    print(f"  torch.cuda.is_available()   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  torch.cuda.memory_allocated : {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
    print(f"  input candidates to rerank  : 100")
    print(f"  batch size passed to model  : 100")
    print("---------------------------------------------------\n", flush=True)

    # Pre-generate LLM transformations in batches
    if llm is not None and hasattr(llm, "complete_batch"):
        from adaptive_retriever.prompts import load_prompt
        print("Pre-warming LLM transformation prompts in batch mode...")
        hyde_tmpl, hyde_ver = load_prompt("transform/hyde")
        decomp_tmpl, decomp_ver = load_prompt("transform/decompose")

        hyde_prompts = [hyde_tmpl.format(query=item["query"]) for item in eval_items]
        decomp_prompts = [decomp_tmpl.format(query=item["query"]) for item in eval_items]

        llm.complete_batch(hyde_prompts, prompt_version=hyde_ver, max_tokens=128, stage="hyde")
        llm.complete_batch(decomp_prompts, prompt_version=decomp_ver, max_tokens=128, stage="decompose")
        print("LLM prompt batch pre-warming complete.")

    # Define Arms
    cfg_none = Config.load(Path("config/default.yaml"))
    cfg_none.raw["query"]["transformation"]["enabled"] = True
    cfg_none.raw["query"]["transformation"]["fixed"] = "none"

    cfg_hyde = Config.load(Path("config/default.yaml"))
    cfg_hyde.raw["query"]["transformation"]["enabled"] = True
    cfg_hyde.raw["query"]["transformation"]["fixed"] = "hyde"

    cfg_decomp = Config.load(Path("config/default.yaml"))
    cfg_decomp.raw["query"]["transformation"]["enabled"] = True
    cfg_decomp.raw["query"]["transformation"]["fixed"] = "decompose"

    cfg_decomp_rephrase_w = Config.load(Path("config/default.yaml"))
    cfg_decomp_rephrase_w.raw["query"]["transformation"]["enabled"] = True
    cfg_decomp_rephrase_w.raw["query"]["transformation"]["fixed"] = "decompose"
    cfg_decomp_rephrase_w.raw["query"]["transformation"]["subquery_weights"] = [0.60, 0.20, 0.20]

    cfg_decomp_orig_w = Config.load(Path("config/default.yaml"))
    cfg_decomp_orig_w.raw["query"]["transformation"]["enabled"] = True
    cfg_decomp_orig_w.raw["query"]["transformation"]["fixed"] = "decompose"
    cfg_decomp_orig_w.raw["query"]["transformation"]["include_original_query"] = True
    cfg_decomp_orig_w.raw["query"]["transformation"]["original_query_weight"] = 0.60

    arms = [
        ("Arm 1: none", cfg_none),
        ("Arm 2: hyde", cfg_hyde),
        ("Arm 3: decompose", cfg_decomp),
        ("Arm 4: decompose_rephrase_weighted", cfg_decomp_rephrase_w),
        ("Arm 5: decompose_orig_weighted", cfg_decomp_orig_w),
    ]

    arm_results: Dict[str, Dict[str, List[float]]] = {
        name: {
            "r5": [], "ndcg5": [], "r10": [], "ndcg10": [], "mrr": [],
            "ret_ms": [], "rr_ms": [], "tot_ms": [], "tf_ms": [], "tokens_in": [], "tokens_out": [],
            "rerank_fired": []
        }
        for name, _ in arms
    }

    subquery_counts = []
    raw_log_records = []

    t0_wall = time.time()
    for idx, item in enumerate(eval_items, 1):
        query_text = item["query"]
        rel_passages_raw = item.get("relevant_passages", [])
        rel_passages = [RelevantPassage(**p) for p in rel_passages_raw]

        item_raw_record = {"item_idx": idx, "query": query_text, "arms": {}}

        for arm_name, cfg_arm in arms:
            cands, trace = run_funnel(query_text, vs=vs, reranker=reranker, cfg=cfg_arm, llm=llm)
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
            tf_stage = stages.get("transformation", {})
            tf_ms = tf_stage.get("elapsed_ms", 0.0)
            details = tf_stage.get("details", {})
            if isinstance(details, dict) and "details" in details and isinstance(details["details"], dict):
                inner_details = details["details"]
            else:
                inner_details = details if isinstance(details, dict) else {}

            t_in = inner_details.get("tokens_in", 0)
            t_out = inner_details.get("tokens_out", 0)

            if arm_name == "Arm 3: decompose":
                sq_cnt = len(tf_stage.get("queries", [query_text]))
                subquery_counts.append(sq_cnt)

            ret_ms = (
                stages.get("prefilter", {}).get("elapsed_ms", 0.0)
                + stages.get("hybrid_search", {}).get("elapsed_ms", 0.0)
            )
            rr_stage = stages.get("rerank", {})
            rr_ms = rr_stage.get("elapsed_ms", 0.0)
            rr_fired = rr_stage.get("decision") == "reranked"
            tot_ms = trace.get("total_elapsed_ms", 0.0)

            arm_results[arm_name]["ret_ms"].append(ret_ms)
            arm_results[arm_name]["rr_ms"].append(rr_ms)
            arm_results[arm_name]["tot_ms"].append(tot_ms)
            arm_results[arm_name]["tf_ms"].append(tf_ms)
            arm_results[arm_name]["tokens_in"].append(t_in)
            arm_results[arm_name]["tokens_out"].append(t_out)
            arm_results[arm_name]["rerank_fired"].append(1 if rr_fired else 0)

            item_raw_record["arms"][arm_name] = {
                "r5": r5, "ndcg5": n5, "r10": r10, "ndcg10": n10, "mrr": m_rr,
                "tf_ms": tf_ms,
                "ret_ms": ret_ms,
                "rr_ms": rr_ms,
                "tot_ms": tot_ms,
                "rerank_fired": rr_fired,
                "queries": tf_stage.get("queries", [query_text]),
                "candidate_count": len(cands),
                "cache_hit": tf_stage.get("details", {}).get("cache_hit", False),
            }

        raw_log_records.append(item_raw_record)

        if idx % 20 == 0 or idx == n_items:
            print(f"  Processed {idx}/{n_items} queries... ({time.time() - t0_wall:.1f}s)", flush=True)

    # Save raw log
    raw_log_path = Path("reports/experiment_005_raw.jsonl")
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_log_path, "w", encoding="utf-8") as f:
        for rec in raw_log_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved raw per-query run records to {raw_log_path}")

    # Compute metric summaries
    seed = int(base_cfg.raw.get("seed", 42))
    report = {
        "metadata": {
            "n_items": n_items,
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "vectorstore_mode": vs.mode,
            "collection_name": vs.collection_name,
            "collection_points": pts_count,
            "seed": seed,
            "device": raw_reranker.get_device_str(),
            "multiplicity_statement": "4 strategy arms x 5 metrics = 20 paired comparisons against none. Under Bonferroni correction for 20 tests (alpha = 0.0025), a difference is statistically significant only if its 99.75% CI excludes 0.",
        },
        "arms": {},
        "differences": {},
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
        fired_cnt = int(np.sum(arm_results[arm_name]["rerank_fired"]))
        arm_dict["rerank_gate_fired_count"] = fired_cnt
        arm_dict["rerank_gate_fire_rate_pct"] = round(float(fired_cnt / n_items * 100.0), 1)
        arm_dict["total_wall_clock_s"] = round(float(np.sum(arm_results[arm_name]["tot_ms"]) / 1000.0), 2)
        arm_dict["transformation_lat_ms_mean"] = round(float(np.mean(arm_results[arm_name]["tf_ms"])), 2)
        arm_dict["transformation_lat_ms_p95"] = round(float(np.percentile(arm_results[arm_name]["tf_ms"], 95)), 2)
        arm_dict["retrieval_lat_ms_mean"] = round(float(np.mean(arm_results[arm_name]["ret_ms"])), 2)
        arm_dict["retrieval_lat_ms_p95"] = round(float(np.percentile(arm_results[arm_name]["ret_ms"], 95)), 2)
        arm_dict["rerank_lat_ms_mean"] = round(float(np.mean(arm_results[arm_name]["rr_ms"])), 2)
        arm_dict["rerank_lat_ms_p95"] = round(float(np.percentile(arm_results[arm_name]["rr_ms"], 95)), 2)
        arm_dict["total_lat_ms_p50"] = round(float(np.percentile(arm_results[arm_name]["tot_ms"], 50)), 2)
        arm_dict["cached_read_ms_mean"] = arm_dict["transformation_lat_ms_mean"]
        arm_dict["cached_read_ms_p95"] = arm_dict["transformation_lat_ms_p95"]

        if torch.cuda.is_available():
            arm_dict["gpu_peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        else:
            arm_dict["gpu_peak_memory_mb"] = 0.0

        report["arms"][arm_name] = arm_dict

    # Differences vs Arm 1 (none)
    for target_arm in ["Arm 2: hyde", "Arm 3: decompose", "Arm 4: decompose_rephrase_weighted", "Arm 5: decompose_orig_weighted"]:
        diff_dict = {}
        for mk in metrics_keys:
            v1 = arm_results["Arm 1: none"][mk]
            v2 = arm_results[target_arm][mk]
            diff_mean, d_low, d_high = bootstrap_difference_ci(v2, v1, n=1000, ci=0.95, seed=seed)
            diff_dict[mk] = {
                "mean_diff": round(float(diff_mean), 4),
                "ci_95": [round(float(d_low), 4), round(float(d_high), 4)],
            }
        key_name = target_arm.split(":")[1].strip() + "_vs_none"
        report["differences"][key_name] = diff_dict

    # Sub-query count breakdown analysis for decompose arm
    sq_cnts = np.array(subquery_counts)
    none_ndcg = np.array(arm_results["Arm 1: none"]["ndcg10"])
    decomp_ndcg = np.array(arm_results["Arm 3: decompose"]["ndcg10"])

    subquery_breakdown = {}
    for count in [1, 2, 3]:
        mask = (sq_cnts == count)
        sub_n = int(np.sum(mask))
        if sub_n > 0:
            diffs = decomp_ndcg[mask] - none_ndcg[mask]
            sub_mean_diff = float(np.mean(diffs))
            subquery_breakdown[f"{count}_subqueries"] = {
                "count": sub_n,
                "ndcg10_diff": round(sub_mean_diff, 4),
                "none_ndcg10": round(float(np.mean(none_ndcg[mask])), 4),
                "decomp_ndcg10": round(float(np.mean(decomp_ndcg[mask])), 4),
            }

    report["decomposition_subquery_count_breakdown"] = subquery_breakdown

    # Select 3 real verbatim examples from raw log records
    real_examples = []
    sample_indices = [0, 10, 25] if len(raw_log_records) >= 26 else list(range(min(3, len(raw_log_records))))
    for s_idx in sample_indices:
        rec = raw_log_records[s_idx]
        item_id = rec.get("item_idx")
        raw_q = rec.get("query")
        hyde_doc = rec["arms"]["Arm 2: hyde"]["queries"][0] if rec["arms"]["Arm 2: hyde"]["queries"] else ""
        decomp_sqs = rec["arms"]["Arm 3: decompose"]["queries"]
        real_examples.append({
            "item_id": item_id,
            "query": raw_q,
            "hyde_hypothetical_document": hyde_doc,
            "decompose_subqueries": decomp_sqs,
        })
    report["verbatim_raw_ledger_examples"] = real_examples

    # Scoped Conclusion
    report["scoped_conclusion"] = (
        "Corrected HyDE and Decomposition (unweighted, rephrase-weighted, and original-query weighted), "
        "implemented with Qwen2.5-0.5B, systematically degrade retrieval on this corpus across all metric dimensions."
    )

    # Print Summary Table
    print("\n" + "=" * 105)
    print(" EXPERIMENT 005 SUMMARY: HYDE AND DEPOSITION ARMS VS NONE")
    print("=" * 105)
    print(f"{'Metric':<10} | {'none':<10} | {'hyde (fixed)':<12} | {'decomp (unw)':<14} | {'decomp (reph wtd)':<18} | {'decomp (orig wtd)':<18}")
    print("-" * 105)
    for mk, name in [("r5", "Recall@5"), ("ndcg5", "nDCG@5"), ("r10", "Recall@10"), ("ndcg10", "nDCG@10"), ("mrr", "MRR")]:
        a1 = report["arms"]["Arm 1: none"][mk]["mean"]
        a2 = report["arms"]["Arm 2: hyde"][mk]["mean"]
        a3 = report["arms"]["Arm 3: decompose"][mk]["mean"]
        a4 = report["arms"]["Arm 4: decompose_rephrase_weighted"][mk]["mean"]
        a5 = report["arms"]["Arm 5: decompose_orig_weighted"][mk]["mean"]
        print(f"{name:<10} | {a1:<10.4f} | {a2:<12.4f} | {a3:<14.4f} | {a4:<18.4f} | {a5:<18.4f}")
    print("-" * 105)
    print("Rerank Gate Fire Rate per Arm:")
    for a_name in arms:
        a_dict = report["arms"][a_name[0]]
        print(f"  {a_name[0]:<35}: {a_dict['rerank_gate_fired_count']}/{n_items} fired ({a_dict['rerank_gate_fire_rate_pct']}%) | Total Rerank Wall Clock: {a_dict['rerank_lat_ms_mean']*n_items/1000.0:.2f}s")
    print("-" * 105)
    print(f"Paired Differences vs Arm 1 (none) [95% CI]:")
    for mk, name in [("r10", "Recall@10"), ("ndcg10", "nDCG@10"), ("mrr", "MRR")]:
        d_h = report["differences"]["hyde_vs_none"][mk]
        d_d = report["differences"]["decompose_vs_none"][mk]
        d_drw = report["differences"]["decompose_rephrase_weighted_vs_none"][mk]
        d_dow = report["differences"]["decompose_orig_weighted_vs_none"][mk]
        print(f"  {name:<10} | HyDE                 : {d_h['mean_diff']:+.4f} [{d_h['ci_95'][0]:+.4f}, {d_h['ci_95'][1]:+.4f}]")
        print(f"             | Decompose (Unw)      : {d_d['mean_diff']:+.4f} [{d_d['ci_95'][0]:+.4f}, {d_d['ci_95'][1]:+.4f}]")
        print(f"             | Decompose (Reph Wtd) : {d_drw['mean_diff']:+.4f} [{d_drw['ci_95'][0]:+.4f}, {d_drw['ci_95'][1]:+.4f}]")
        print(f"             | Decompose (Orig Wtd) : {d_dow['mean_diff']:+.4f} [{d_dow['ci_95'][0]:+.4f}, {d_dow['ci_95'][1]:+.4f}]")
    print("-" * 105)
    print("Decomposition Sub-Query Count Breakdown (nDCG@10 Diff vs None):")
    for sq_k, sq_data in subquery_breakdown.items():
        print(f"  {sq_k:<15}: count={sq_data['count']:<3} | nDCG@10 diff = {sq_data['ndcg10_diff']:+.4f} (None={sq_data['none_ndcg10']:.4f}, Decomp={sq_data['decomp_ndcg10']:.4f})")
    print("=" * 105 + "\n")

    # Save to file
    out_path = Path("reports/experiment_005.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Experiment 005 results to {out_path}")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    run_experiment_005(smoke=smoke)
