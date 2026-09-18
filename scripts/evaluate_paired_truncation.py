"""Phase 5 Final Evaluation: Paired Comparison Under Truncation-Aware Scoring (Task T-02g).

Evaluates baseline vs. shipped retrieval systems under full-text and first 2,000-character scoring,
computes paired bootstrap CIs (10,000 resamples), two-sided p-values, and subset decomposition.
"""

import json
from pathlib import Path
import numpy as np
from adaptive_retriever.config import Config
from adaptive_retriever.vectorstore import VectorStore
from adaptive_retriever.funnel.baseline import run_baseline
from adaptive_retriever.funnel.pipeline import run_funnel
from adaptive_retriever.funnel.rerank import Reranker
from adaptive_retriever.models import EvalItem
from adaptive_retriever.eval.metrics import (
    recall_at_k,
    ndcg_at_k,
    mrr,
    is_relevant,
    normalise
)

def find_gold_offset(chunk_text: str, passage_text: str) -> int:
    idx = chunk_text.find(passage_text)
    if idx != -1:
        return idx
    norm_pass = normalise(passage_text)
    norm_chunk = normalise(chunk_text[:50000])
    words = norm_pass.split()
    if len(words) >= 8:
        ngram = " ".join(words[:8])
        pos_norm = norm_chunk.find(ngram)
        if pos_norm != -1:
            frac = pos_norm / max(1, len(norm_chunk))
            return int(frac * min(len(chunk_text), 50000))
    for w in words[:5]:
        if len(w) > 4:
            p = chunk_text.lower().find(w)
            if p != -1:
                return p
    return 0

def bootstrap_paired(ad_vals, base_vals, n=10000, seed=42):
    ad_arr = np.array(ad_vals, dtype=float)
    base_arr = np.array(base_vals, dtype=float)
    diff_arr = ad_arr - base_arr
    
    ad_mean = float(np.mean(ad_arr))
    base_mean = float(np.mean(base_arr))
    diff_mean = float(np.mean(diff_arr))
    
    rng = np.random.default_rng(seed)
    n_samples = len(diff_arr)
    boot_diffs = []
    
    for _ in range(n):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_diffs.append(np.mean(diff_arr[idx]))
        
    lower = float(np.percentile(boot_diffs, 2.5))
    upper = float(np.percentile(boot_diffs, 97.5))
    
    if diff_mean > 0:
        p_val = float(np.mean(np.array(boot_diffs) <= 0)) * 2.0
    elif diff_mean < 0:
        p_val = float(np.mean(np.array(boot_diffs) >= 0)) * 2.0
    else:
        p_val = 1.0
    p_val = min(1.0, max(0.0, p_val))
    
    return base_mean, ad_mean, diff_mean, lower, upper, p_val

def main():
    cfg = Config.load(Path("config/default.yaml"))
    vs = VectorStore.from_config(cfg)
    reranker = Reranker(device="auto")
    
    eval_path = Path("data/eval/eval_set.jsonl")
    with open(eval_path, "r", encoding="utf-8") as f:
        items_raw = [json.loads(line) for line in f if line.strip()]
    eval_items = [EvalItem(**item) for item in items_raw if item.get("provenance") != "synthetic_rejected"]
    
    print(f"Loaded {len(eval_items)} valid eval items.")
    
    with open("reports/candidate_pools_cache.json", "r", encoding="utf-8") as f:
        cand_cache = json.load(f)

    affected_item_indices = set()
    for idx, item in enumerate(eval_items):
        q = item.query
        gold_passages = item.relevant_passages
        cands_raw = cand_cache.get(q, [])
        item_affected = False
        
        for p_obj in gold_passages:
            p_text = p_obj.passage_text
            for c in cands_raw:
                c_text = c.get("text", "")
                if is_relevant(c_text, p_obj):
                    c_offset = find_gold_offset(c_text, p_text)
                    if c_offset >= 2000:
                        item_affected = True
                        break
            if item_affected:
                break
        if item_affected:
            affected_item_indices.add(idx)

    base_full = {"r10": [], "ndcg10": [], "mrr": []}
    ad_full = {"r10": [], "ndcg10": [], "mrr": []}
    
    base_2k = {"r10": [], "ndcg10": [], "mrr": []}
    ad_2k = {"r10": [], "ndcg10": [], "mrr": []}
    
    for idx, item in enumerate(eval_items):
        cands_base, _ = run_baseline(item.query, vs=vs, cfg=cfg, limit=10)
        cands_ad, _ = run_funnel(item.query, vs=vs, cfg=cfg, reranker=reranker)
        
        texts_base = [c.text for c in cands_base]
        texts_ad = [c.text for c in cands_ad]
        
        texts_base_2k = [c.text[:2000] for c in cands_base]
        texts_ad_2k = [c.text[:2000] for c in cands_ad]
        
        base_full["r10"].append(recall_at_k(texts_base, item.relevant_passages, 10))
        base_full["ndcg10"].append(ndcg_at_k(texts_base, item.relevant_passages, 10))
        base_full["mrr"].append(mrr(texts_base, item.relevant_passages))
        
        ad_full["r10"].append(recall_at_k(texts_ad, item.relevant_passages, 10))
        ad_full["ndcg10"].append(ndcg_at_k(texts_ad, item.relevant_passages, 10))
        ad_full["mrr"].append(mrr(texts_ad, item.relevant_passages))
        
        base_2k["r10"].append(recall_at_k(texts_base_2k, item.relevant_passages, 10))
        base_2k["ndcg10"].append(ndcg_at_k(texts_base_2k, item.relevant_passages, 10))
        base_2k["mrr"].append(mrr(texts_base_2k, item.relevant_passages))
        
        ad_2k["r10"].append(recall_at_k(texts_ad_2k, item.relevant_passages, 10))
        ad_2k["ndcg10"].append(ndcg_at_k(texts_ad_2k, item.relevant_passages, 10))
        ad_2k["mrr"].append(mrr(texts_ad_2k, item.relevant_passages))

    report = {
        "multiplicity": "3 hypothesis tests (Recall@10, nDCG@10, MRR). Bonferroni alpha = 0.0167.",
        "metrics": {},
        "subset_decomposition": {}
    }
    
    for metric in ["r10", "ndcg10", "mrr"]:
        b_f, a_f, d_f, l_f, u_f, p_f = bootstrap_paired(ad_full[metric], base_full[metric])
        b_2k, a_2k, d_2k, l_2k, u_2k, p_2k = bootstrap_paired(ad_2k[metric], base_2k[metric])
        report["metrics"][metric] = {
            "full_text": {"baseline": b_f, "shipped": a_f, "diff": d_f, "ci_95": [l_f, u_f], "p_value": p_f},
            "first_2k_chars": {"baseline": b_2k, "shipped": a_2k, "diff": d_2k, "ci_95": [l_2k, u_2k], "p_value": p_2k}
        }

    affected_indices = sorted(list(affected_item_indices))
    unaffected_indices = [i for i in range(len(eval_items)) if i not in affected_item_indices]
    
    report["subset_decomposition"] = {
        "affected_count": len(affected_indices),
        "unaffected_count": len(unaffected_indices),
        "affected_recall10": {
            "full_text": {"baseline": float(np.mean([base_full["r10"][i] for i in affected_indices])), "shipped": float(np.mean([ad_full["r10"][i] for i in affected_indices]))},
            "first_2k_chars": {"baseline": float(np.mean([base_2k["r10"][i] for i in affected_indices])), "shipped": float(np.mean([ad_2k["r10"][i] for i in affected_indices]))}
        },
        "unaffected_recall10": {
            "full_text": {"baseline": float(np.mean([base_full["r10"][i] for i in unaffected_indices])), "shipped": float(np.mean([ad_full["r10"][i] for i in unaffected_indices]))},
            "first_2k_chars": {"baseline": float(np.mean([base_2k["r10"][i] for i in unaffected_indices])), "shipped": float(np.mean([ad_2k["r10"][i] for i in unaffected_indices]))}
        }
    }
    
    out_file = Path("reports/experiment_001_paired_truncation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {out_file}")

if __name__ == "__main__":
    main()
