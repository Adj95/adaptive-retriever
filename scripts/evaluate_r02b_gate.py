#!/usr/bin/env python3
"""Task R-02b: Gate Signal Evaluation (Cascade Router Gate Decision).

Evaluates candidate signals S1-S4 on Tier 1 configuration exactly as committed by R-03.
Saves report to reports/r02b_gate.json.
"""

import sys
import os
import time
import json
import logging
import re
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import is_relevant
from adaptive_retriever.funnel.pipeline import run_funnel, warm_up
from adaptive_retriever.funnel.rerank import Reranker
from adaptive_retriever.llm import get_llm_client
from adaptive_retriever.models import EvalItem
from adaptive_retriever.vectorstore import VectorStore
from scripts.evaluate_r03_tiers import build_tier_config

logging.basicConfig(level=logging.WARNING)

EVAL_PATH = Path("data/eval/eval_set.jsonl")
REPORT_PATH = Path("reports/r02b_gate.json")

STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "what", "which", "who", "whom", "this", "that", "these", "those"}

def extract_content_words(text: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]

def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 10]

def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC score manually to avoid external dependencies if needed."""
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank scores
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1
    
    # Handle ties by averaging ranks
    unique_scores, inverse_indices, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            idx = np.where(inverse_indices == i)[0]
            ranks[idx] = np.mean(ranks[idx])

    sum_pos_ranks = np.sum(ranks[pos_mask])
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def compute_bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, resamples: int = 10000, seed: int = 42) -> tuple[float, list[float]]:
    np.random.seed(seed)
    n = len(y_true)
    auc_obs = compute_roc_auc(y_true, y_score)
    resample_aucs = []
    
    for _ in range(resamples):
        idx = np.random.randint(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        resample_aucs.append(compute_roc_auc(yt, ys))
        
    low = float(np.percentile(resample_aucs, 2.5))
    high = float(np.percentile(resample_aucs, 97.5))
    return auc_obs, [low, high]

def has_8gram_match(sentence: str, context: str) -> bool:
    s_words = extract_content_words(sentence)
    if len(s_words) < 8:
        # Check raw word n-grams
        raw_words = re.findall(r"\b\w+\b", sentence.lower())
        if len(raw_words) < 8:
            return sentence.lower() in context.lower()
        for i in range(len(raw_words) - 7):
            gram = " ".join(raw_words[i:i+8])
            if gram in context.lower():
                return True
        return False
    else:
        for i in range(len(s_words) - 7):
            gram = " ".join(s_words[i:i+8])
            if gram in context.lower():
                return True
        return False

def recompute_shuffled_null(vs: VectorStore, num_pairs: int = 500) -> float:
    print("Re-computing shuffled null cosine similarity threshold on current index...", flush=True)
    res, _ = vs.scroll(limit=50, with_payload=True)
    texts = [p.payload.get("text", "") for p in res if p.payload and p.payload.get("text")]
    
    if len(texts) < 10:
        return 0.56

    np.random.seed(42)
    pairs = []
    embedder = vs.embedding_client
    
    for _ in range(num_pairs):
        i1, i2 = np.random.choice(len(texts), size=2, replace=False)
        s1_list = split_sentences(texts[i1])
        s2_list = split_sentences(texts[i2])
        if s1_list and s2_list:
            st1 = str(np.random.choice(s1_list))
            st2 = str(np.random.choice(s2_list))
            pairs.append((st1, st2))

    if not pairs:
        return 0.56

    all_s1 = [p[0] for p in pairs]
    all_s2 = [p[1] for p in pairs]
    v1 = np.array(embedder.encode_batch(all_s1))
    v2 = np.array(embedder.encode_batch(all_s2))
    
    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
    cosines = np.sum(v1_norm * v2_norm, axis=1)
            
    null_p95 = float(np.percentile(cosines, 95))
    print(f"Recomputed shuffled null cosine 95th percentile: {null_p95:.4f}", flush=True)
    return null_p95

def evaluate_r02b_gate():
    base_cfg = Config.load("config/default.yaml")
    if "models" in base_cfg.raw and "llm" in base_cfg.raw["models"]:
        base_cfg.raw["models"]["llm"]["device"] = "auto"

    t1_cfg = build_tier_config(base_cfg, 1)

    print("=" * 80)
    print("TASK R-02b: GATE SIGNAL EVALUATION")
    print(f"Python Executable: {sys.executable}")
    import torch
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    print(f"CUDA Version: {cuda_ver}")
    print("LOADED TIER 1 CONFIGURATION CONFIRMATION:")
    print(f"  funnel.hybrid.dense_limit : {t1_cfg.raw['funnel']['hybrid']['dense_limit']}")
    print(f"  funnel.hybrid.weights     : {t1_cfg.raw['funnel']['hybrid']['weights']}")
    print(f"  funnel.prefilter.kind     : {t1_cfg.raw['funnel']['prefilter']['kind_filter']}")
    print(f"  funnel.rerank.mode        : {t1_cfg.raw['funnel']['rerank']['mode']}")
    print(f"  generation.context_top_k  : {t1_cfg.raw['generation']['context_top_k']}")
    print("=" * 80, flush=True)

    vs = VectorStore.from_config(t1_cfg)
    reranker = Reranker(device="auto")
    llm = get_llm_client(t1_cfg)
    warm_up(vs, reranker)

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        all_eval_items = [EvalItem(**json.loads(line)) for line in f if line.strip()]

    valid_eval_items = [item for item in all_eval_items if item.provenance != "synthetic_rejected"]
    N = len(valid_eval_items)
    print(f"Loaded {N} valid evaluation queries (N=169).\n", flush=True)

    # Recompute shuffled null cosine
    shuffled_null_p95 = recompute_shuffled_null(vs, num_pairs=300)

    labels = []
    s1_vals, s1_costs = [], []
    s2_vals, s2_costs = [], []
    s3_vals, s3_costs = [], []
    s4_vals, s4_costs = [], []

    prompts = []
    contexts_top3 = []

    print("Step 1: Running Tier 1 retrieval & computing S1, S2 for 169 queries...", flush=True)
    for i, item in enumerate(valid_eval_items, 1):
        query = item.query
        gold_passages = item.relevant_passages

        # B1: Run Tier 1 and record label (gold passage present in top-5 candidates)
        cands, _ = run_funnel(query, vs=vs, cfg=t1_cfg, reranker=reranker, llm=None)
        cands_top5 = cands[:5]
        label = 1 if any(any(is_relevant(c.text, pas) for pas in gold_passages) for c in cands_top5) else 0
        labels.append(label)

        # B2 S1: rrf_margin
        t0_s1 = time.perf_counter()
        scores = [c.dense_score if c.dense_score is not None else c.fused_score for c in cands_top5]
        if len(scores) >= 2 and scores[0] > 0:
            s1 = float((scores[0] - scores[1]) / scores[0])
        else:
            s1 = 0.0
        t1_s1 = time.perf_counter()
        s1_vals.append(s1)
        s1_costs.append((t1_s1 - t0_s1) * 1000.0)

        # B2 S2: query_overlap
        t0_s2 = time.perf_counter()
        q_words = extract_content_words(query)
        context_top3_text = " ".join([c.text for c in cands_top5[:3]])
        if q_words:
            c_lower = context_top3_text.lower()
            s2 = float(sum(1 for w in q_words if w in c_lower) / len(q_words))
        else:
            s2 = 0.0
        t1_s2 = time.perf_counter()
        s2_vals.append(s2)
        s2_costs.append((t1_s2 - t0_s2) * 1000.0)

        contexts_top3.append(context_top3_text)
        prompts.append(f"Context:\n{context_top3_text}\n\nQuestion: {query}\nAnswer:")

        if i % 25 == 0 or i == N:
            print(f"[Step 1 {i}/{N}] Processed item {item.item_id}: label={label}, s1={s1:.4f}, s2={s2:.4f}", flush=True)

    print("Step 2: Batch generating LLM responses for all 169 queries...", flush=True)
    if hasattr(llm, "batch_size"):
        llm.batch_size = 4
    t0_gen = time.perf_counter()
    llm_resps = llm.complete_batch(prompts, system="Answer the question directly based on the context.", max_tokens=64)
    t1_gen = time.perf_counter()
    total_gen_ms = (t1_gen - t0_gen) * 1000.0
    per_item_gen_ms = total_gen_ms / N
    print(f"Batch generation completed in {total_gen_ms/1000.0:.2f}s ({per_item_gen_ms:.2f} ms/item average).", flush=True)

    print("Step 3: Computing S3 (attribution_8gram) and S4 (attribution_2stage)...", flush=True)
    for i in range(N):
        context_top3_text = contexts_top3[i]
        resp_text = llm_resps[i].text
        ans_sents = split_sentences(resp_text)

        # S3: attribution_8gram
        if ans_sents:
            m8_cnt = sum(1 for sent in ans_sents if has_8gram_match(sent, context_top3_text))
            s3 = float(m8_cnt / len(ans_sents))
        else:
            s3 = 0.0
        s3_vals.append(s3)
        s3_costs.append(per_item_gen_ms)

        # S4: attribution_2stage
        t0_s4 = time.perf_counter()
        ctx_sents = split_sentences(context_top3_text)
        grounded_cnt = 0

        if ans_sents:
            unmatched_sents = []
            for sent in ans_sents:
                if has_8gram_match(sent, context_top3_text):
                    grounded_cnt += 1
                else:
                    unmatched_sents.append(sent)

            if unmatched_sents and ctx_sents:
                all_to_encode = unmatched_sents + ctx_sents[:10]
                all_vecs = np.array(vs.embedding_client.encode_batch(all_to_encode))
                s_vecs = all_vecs[:len(unmatched_sents)]
                c_vecs = all_vecs[len(unmatched_sents):]
                
                s_norm = s_vecs / (np.linalg.norm(s_vecs, axis=1, keepdims=True) + 1e-9)
                c_norm = c_vecs / (np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-9)
                
                cos_matrix = np.dot(s_norm, c_norm.T)
                max_cosines = np.max(cos_matrix, axis=1)
                
                grounded_cnt += int(np.sum(max_cosines > shuffled_null_p95))

            s4 = float(grounded_cnt / len(ans_sents))
        else:
            s4 = 0.0

        t1_s4 = time.perf_counter()
        emb_ms = (t1_s4 - t0_s4) * 1000.0
        s4_vals.append(s4)
        s4_costs.append(per_item_gen_ms + emb_ms)

    y_true = np.array(labels)
    pos_rate = float(np.mean(y_true))
    print(f"\nPositive Label Rate (Gold passage in Tier 1 top-5): {pos_rate:.4f} ({np.sum(y_true)}/{N})", flush=True)

    # Shuffled control labels
    np.random.seed(42)
    y_shuffled = np.random.permutation(y_true)

    signals = {
        "S1 (rrf_margin)": (np.array(s1_vals), s1_costs),
        "S2 (query_overlap)": (np.array(s2_vals), s2_costs),
        "S3 (attribution_8gram)": (np.array(s3_vals), s3_costs),
        "S4 (attribution_2stage)": (np.array(s4_vals), s4_costs),
    }

    print("\n" + "=" * 105)
    print("B3 EVALUATION OF CANDIDATE SIGNALS (Comparison Count = 4 Candidate Signals vs Control)")
    print(f"{'Signal':<25} | {'ROC AUC [95% CI]':<22} | {'Shuffled Control AUC':<20} | {'Mean Cost (ms)':<15} | {'AUC / 100ms':<12}")
    print("-" * 105)

    gate_table = []
    best_auc = 0.0
    best_signal_name = ""
    best_ci = [0.0, 0.0]
    best_auc_per_cost = 0.0

    for s_name, (s_vals, s_c_list) in signals.items():
        auc_obs, ci = compute_bootstrap_auc(y_true, s_vals, resamples=10000, seed=42)
        auc_shuf = compute_roc_auc(y_shuffled, s_vals)
        mean_cost = float(np.mean(s_c_list))
        p95_cost = float(np.percentile(s_c_list, 95))
        
        # Safe AUC per 100ms (floor cost at 0.1ms for 0ms signals)
        effective_cost_100ms = max(0.001, mean_cost / 100.0)
        auc_per_cost = float(auc_obs / effective_cost_100ms)

        ci_str = f"{auc_obs:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        shuf_str = f"{auc_shuf:.4f}"
        
        row = {
            "signal": s_name,
            "auc": ci_str,
            "shuffled_control_auc": shuf_str,
            "mean_cost_ms": round(mean_cost, 2),
            "p95_cost_ms": round(p95_cost, 2),
            "auc_per_100ms": round(auc_per_cost, 4),
            "auc_val": auc_obs,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
        gate_table.append(row)
        print(f"{s_name:<25} | {ci_str:<22} | {shuf_str:<20} | {mean_cost:<15.2f} | {auc_per_cost:<12.4f}")

        if auc_obs > best_auc:
            best_auc = auc_obs
            best_signal_name = s_name
            best_ci = ci
            best_auc_per_cost = auc_per_cost

    print("=" * 105 + "\n")

    # B4: GATE VERDICT
    query_confidence_auc = 0.6124
    query_confidence_ci = [0.5281, 0.6954]

    verdict_code = ""
    verdict_msg = ""

    if best_auc >= 0.70 and best_ci[0] > 0.50:
        verdict_code = "(a)"
        verdict_msg = f"PROCEED. Best signal is {best_signal_name} (ROC AUC {best_auc:.4f} [{best_ci[0]:.4f}, {best_ci[1]:.4f}], AUC per 100ms = {best_auc_per_cost:.4f})."
    elif 0.60 <= best_auc < 0.70:
        verdict_code = "(b)"
        verdict_msg = f"PROCEED but the router must be labelled WEAKLY DISCRIMINATING everywhere it is reported. Best signal is {best_signal_name} (ROC AUC {best_auc:.4f} [{best_ci[0]:.4f}, {best_ci[1]:.4f}])."
    else:
        verdict_code = "(c)"
        verdict_msg = f"DO NOT BUILD THE ROUTER. Best signal ROC AUC is {best_auc:.4f} [{best_ci[0]:.4f}, {best_ci[1]:.4f}] (< 0.60 or CI includes 0.50). The project reports the oracle bound and the null instead."

    beats_query_side = best_auc > query_confidence_auc and best_ci[0] > query_confidence_ci[0]
    
    print("=" * 80)
    print(f"GATE VERDICT: {verdict_code} - {verdict_msg}")
    if not beats_query_side:
        print(f"COMPARISON NOTE: None of S1-S4 convincingly beats the query-side confidence score (ROC AUC {query_confidence_auc:.4f} [{query_confidence_ci[0]:.4f}, {query_confidence_ci[1]:.4f}]).")
    print("=" * 80 + "\n")

    report_output = {
        "tier1_config_confirmation": {
            "dense_limit": t1_cfg.raw['funnel']['hybrid']['dense_limit'],
            "weights": t1_cfg.raw['funnel']['hybrid']['weights'],
            "kind_filter": t1_cfg.raw['funnel']['prefilter']['kind_filter'],
            "rerank_mode": t1_cfg.raw['funnel']['rerank']['mode'],
            "context_top_k": t1_cfg.raw['generation']['context_top_k'],
        },
        "positive_rate": pos_rate,
        "shuffled_null_p95_cosine": shuffled_null_p95,
        "comparison_count": 4,
        "query_side_confidence_auc": f"{query_confidence_auc:.4f} [{query_confidence_ci[0]:.4f}, {query_confidence_ci[1]:.4f}]",
        "table": gate_table,
        "verdict_code": verdict_code,
        "verdict": verdict_msg,
        "beats_query_side_confidence": beats_query_side,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_output, f, indent=2)
    print(f"Saved TASK R-02b report to {REPORT_PATH}", flush=True)

    return report_output

if __name__ == "__main__":
    evaluate_r02b_gate()
