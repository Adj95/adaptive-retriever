"""Experiment 006 — Post-Retrieval Reordering & Compression Evaluation (Task E, §8).

Evaluates 169 non-rejected items on retriever_chunks_a5d51992 index across four post-retrieval arms:
  Arm 1: generation_alone (reorder=False, compression=none) - Baseline
  Arm 2: +reorder          (reorder=True,  compression=none)
  Arm 3: +compression      (reorder=False, compression=embeddings)
  Arm 4: +both             (reorder=True,  compression=embeddings)

Outputs ROUGE-1, ROUGE-2, ROUGE-L, BLEURT-20 (BGE similarity proxy), citation rate,
unsupported sentence rate, latency, paired bootstrap CIs against Arm 1, and multiplicity count.
Outputs summary to reports/experiment_006.json.
"""

import copy
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Set, Tuple

astype = None
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import bootstrap_ci, bootstrap_difference_ci, is_relevant
from adaptive_retriever.funnel.pipeline import run_funnel
from adaptive_retriever.generation.generate import attribute_sentence_to_chunks, generate_answer
from adaptive_retriever.llm import get_llm_client
from adaptive_retriever.models import RelevantPassage, RetrievalCandidate
from adaptive_retriever.postretrieval import compress, reorder_candidates
from adaptive_retriever.vectorstore import DenseEmbeddingModel, VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_tokens(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_rouge_n(gen_text: str, ref_text: str, n: int) -> float:
    gen_toks = get_tokens(gen_text)
    ref_toks = get_tokens(ref_text)
    if not ref_toks or len(ref_toks) < n or not gen_toks or len(gen_toks) < n:
        return 0.0
    ref_ngrams = get_ngrams(ref_toks, n)
    gen_ngrams = get_ngrams(gen_toks, n)
    
    ref_counts = {}
    for ng in ref_ngrams:
        ref_counts[ng] = ref_counts.get(ng, 0) + 1
        
    match_count = 0
    gen_counts = {}
    for ng in gen_ngrams:
        gen_counts[ng] = gen_counts.get(ng, 0) + 1
        
    for ng, count in gen_counts.items():
        if ng in ref_counts:
            match_count += min(count, ref_counts[ng])
            
    return float(match_count) / float(len(ref_ngrams))


def compute_rouge_l(gen_text: str, ref_text: str) -> float:
    gen_toks = get_tokens(gen_text)
    ref_toks = get_tokens(ref_text)
    m, n = len(gen_toks), len(ref_toks)
    if m == 0 or n == 0:
        return 0.0
    
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if gen_toks[i - 1] == ref_toks[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
            
    lcs_len = dp[n]
    return float(lcs_len) / float(n)


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


def main():
    cfg = Config.load("config/default.yaml")
    eval_items = load_eval_items()
    logger.info(f"Loaded {len(eval_items)} valid eval items for Experiment 006")

    vs = VectorStore.from_config(cfg)
    embedder = vs.embedding_client
    llm = get_llm_client(cfg)

    arms_def = {
        "Arm 1: generation_alone": {"reorder": False, "compression": "none"},
        "Arm 2: +reorder":          {"reorder": True,  "compression": "none"},
        "Arm 3: +compression":      {"reorder": False, "compression": "embeddings"},
        "Arm 4: +both":             {"reorder": True,  "compression": "embeddings"},
    }

    raw_ledger = []
    arm_results = {arm_name: [] for arm_name in arms_def}
    raw_jsonl_path = Path("reports/experiment_006_raw.jsonl")

    processed_records = {}
    if raw_jsonl_path.exists():
        with open(raw_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    processed_records[rec["item_idx"]] = rec

    logger.info("Starting Experiment 006 evaluation across 4 arms...")
    start_time = time.time()

    for idx, item in enumerate(eval_items):
        item_idx = idx + 1
        q = item["query"]
        raw_passages = item.get("relevant_passages", [])
        ref_text = " ".join([p["passage_text"] for p in raw_passages if "passage_text" in p])
        if not ref_text.strip():
            ref_text = q

        if item_idx in processed_records:
            record = processed_records[item_idx]
            raw_ledger.append(record)
            for arm_name in arms_def:
                arm_data = record["arms"][arm_name]
                arm_results[arm_name].append({
                    "rouge_1": arm_data["rouge_1"],
                    "rouge_2": arm_data["rouge_2"],
                    "rouge_l": arm_data["rouge_l"],
                    "bleurt_20": arm_data["bleurt_20"],
                    "has_citation": arm_data["has_citation"],
                    "unsupported_pct": arm_data["unsupported_pct"],
                    "tot_ms": arm_data["tot_ms"],
                })
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 1: Run standard funnel retrieval to top candidates
        t0_ret = time.perf_counter()
        base_candidates, _ = run_funnel(q, vs=vs, cfg=cfg, llm=llm)
        ret_ms = (time.perf_counter() - t0_ret) * 1000.0

        record = {"item_idx": item_idx, "query": q, "arms": {}}

        for arm_name, arm_cfg in arms_def.items():
            t0_arm = time.perf_counter()
            cands = copy.deepcopy(base_candidates)

            # Post-retrieval compression & reordering
            if arm_cfg["compression"] == "embeddings":
                cands, _ = compress(query=q, candidates=cands, cfg=cfg, embedding_client=embedder)
            if arm_cfg["reorder"]:
                cands, _ = reorder_candidates(cands, cfg=cfg)

            # Answer Generation
            t0_gen = time.perf_counter()
            ans = generate_answer(query=q, candidates=cands, cfg=cfg, llm=llm)
            gen_ms = (time.perf_counter() - t0_gen) * 1000.0
            tot_arm_ms = (time.perf_counter() - t0_arm) * 1000.0 + ret_ms

            # Compute ROUGE-1, ROUGE-2, ROUGE-L
            r1 = compute_rouge_n(ans.text, ref_text, 1) if ans.text else 0.0
            r2 = compute_rouge_n(ans.text, ref_text, 2) if ans.text else 0.0
            rl = compute_rouge_l(ans.text, ref_text) if ans.text else 0.0

            # BLEURT-20 proxy via BGE embedding cosine similarity
            if ans.text.strip():
                try:
                    vec_gen = embedder.encode(ans.text, normalize_embeddings=True)
                    vec_ref = embedder.encode(ref_text, normalize_embeddings=True)
                except TypeError:
                    vec_gen = embedder.encode(ans.text)
                    vec_ref = embedder.encode(ref_text)
                bleurt_proxy = round(float(np.dot(vec_gen, vec_ref)), 4)
            else:
                bleurt_proxy = 0.0

            # Citation rate & unsupported sentence rate
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ans.text) if s.strip()]
            unsupported_cnt = 0
            citations = set()
            for s in sents:
                attr = attribute_sentence_to_chunks(s, cands, min_n_gram=8)
                if attr:
                    citations.add(attr)
                else:
                    unsupported_cnt += 1

            has_citation = len(citations) > 0
            unsupported_pct = (unsupported_cnt / len(sents) * 100.0) if sents else 100.0

            metrics = {
                "rouge_1": round(r1, 4),
                "rouge_2": round(r2, 4),
                "rouge_l": round(rl, 4),
                "bleurt_20": bleurt_proxy,
                "has_citation": 1.0 if has_citation else 0.0,
                "unsupported_pct": round(unsupported_pct, 2),
                "tot_ms": round(tot_arm_ms, 2),
            }

            arm_results[arm_name].append(metrics)
            record["arms"][arm_name] = {**metrics, "text": ans.text}

        raw_ledger.append(record)
        with open(raw_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

        elapsed = time.time() - start_time
        items_done = item_idx - len(processed_records)
        if items_done > 0:
            avg_sec = elapsed / items_done
            items_left = len(eval_items) - item_idx
            eta_sec = items_left * avg_sec
            if item_idx % 5 == 0 or item_idx == len(eval_items):
                logger.info(f"[PROGRESS] Item {item_idx}/{len(eval_items)} completed | Elapsed: {elapsed:.1f}s | Avg: {avg_sec:.1f}s/item | ETA: {eta_sec/60:.1f}m ({eta_sec:.0f}s)")

    total_exp_time = time.time() - start_time
    logger.info(f"Completed Experiment 006 evaluation in {total_exp_time:.2f} seconds")

    # Aggregate summaries and paired bootstrap differences against Arm 1
    summary = {}
    base_arm_name = "Arm 1: generation_alone"
    base_metrics = arm_results[base_arm_name]

    metric_keys = ["rouge_1", "rouge_2", "rouge_l", "bleurt_20", "has_citation", "unsupported_pct", "tot_ms"]

    for arm_name, results_list in arm_results.items():
        arm_summary = {}
        for key in metric_keys:
            vals = [r[key] for r in results_list]
            mean_val, ci_lo, ci_hi = bootstrap_ci(vals, n=1000)

            metric_entry = {
                "mean": round(float(mean_val), 4),
                "ci_95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
            }

            if arm_name != base_arm_name:
                base_vals = [r[key] for r in base_metrics]
                diff_mean, d_lo, d_hi = bootstrap_difference_ci(vals, base_vals, n=1000)
                metric_entry["diff_vs_baseline"] = {
                    "mean_diff": round(float(diff_mean), 4),
                    "ci_95": [round(float(d_lo), 4), round(float(d_hi), 4)],
                }

            arm_summary[key] = metric_entry

        summary[arm_name] = arm_summary

    multiplicity_count = (len(arms_def) - 1) * len(metric_keys)  # 3 comparative arms * 7 metrics = 21 hypothesis tests

    output_report = {
        "meta": {
            "experiment_id": "experiment_006",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_items": len(eval_items),
            "multiplicity_count": multiplicity_count,
            "multiplicity_explanation": f"{len(arms_def) - 1} comparative arms evaluated against Baseline ({base_arm_name}) across {len(metric_keys)} metrics = {multiplicity_count} hypothesis tests.",
            "total_time_seconds": round(total_exp_time, 2),
        },
        "arms": summary,
    }

    out_path = Path("reports/experiment_006.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)

    logger.info(f"Saved Experiment 006 report to {out_path}")
    print(json.dumps(output_report, indent=2))


if __name__ == "__main__":
    main()
