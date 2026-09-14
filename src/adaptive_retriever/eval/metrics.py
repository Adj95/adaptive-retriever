"""Evaluation metrics implemented directly using numpy and scipy (spec §8.3 & Addendum B4)."""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from adaptive_retriever.config import Config
from adaptive_retriever.models import EvalItem, RelevantPassage, RetrievalCandidate


def normalise(text: str) -> str:
    """lowercase, collapse all whitespace to single spaces, strip punctuation
    except intra-word hyphens and decimal points, unicode NFKC."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    cleaned_chars = []
    for i, char in enumerate(text):
        if char.isalnum() or char == " ":
            cleaned_chars.append(char)
        elif (
            char == "-"
            and 0 < i < len(text) - 1
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        ):
            cleaned_chars.append(char)
        elif (
            char == "."
            and 0 < i < len(text) - 1
            and text[i - 1].isdigit()
            and text[i + 1].isdigit()
        ):
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")
    return re.sub(r"\s+", " ", "".join(cleaned_chars)).strip()


def get_ngrams(tokens: List[str], n: int) -> Set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def is_relevant(
    chunk_text: str,
    passage: RelevantPassage,
    instrument_log: Optional[Dict] = None,
) -> bool:
    """A chunk is relevant to a passage when BOTH hold:
    1. token-level containment >= 0.60
    2. contiguity: at least one 8-gram from passage_norm appears verbatim in chunk_norm
    """
    passage_norm = passage.passage_norm or normalise(passage.passage_text)
    chunk_norm = normalise(chunk_text)

    passage_tokens = passage_norm.split()
    chunk_tokens = chunk_norm.split()

    if not passage_tokens:
        return False

    passage_token_set = set(passage_tokens)
    chunk_token_set = set(chunk_tokens)

    overlap_count = len(passage_token_set & chunk_token_set)
    containment = overlap_count / float(len(passage_token_set))
    condition_1 = containment >= 0.60

    n_gram_size = 8
    passage_ngrams = get_ngrams(passage_tokens, n_gram_size)
    chunk_ngrams = get_ngrams(chunk_tokens, n_gram_size)

    condition_2 = bool(passage_ngrams & chunk_ngrams)

    if instrument_log is not None:
        if condition_1 and not condition_2:
            instrument_log["condition_1_pass_condition_2_fail"] = (
                instrument_log.get("condition_1_pass_condition_2_fail", 0) + 1
            )

    return condition_1 and condition_2


def passages_to_chunk_relevance(
    chunk_text: str, relevant_passages: List[RelevantPassage]
) -> bool:
    return any(is_relevant(chunk_text, passage) for passage in relevant_passages)


def recall_at_k(
    retrieved_chunk_texts: List[str],
    relevant_passages: List[RelevantPassage],
    k: int,
) -> float:
    if not relevant_passages:
        return 0.0
    top_k = retrieved_chunk_texts[:k]
    covered = sum(
        1
        for passage in relevant_passages
        if any(is_relevant(chunk, passage) for chunk in top_k)
    )
    return float(covered) / float(len(relevant_passages))


def precision_at_k(
    retrieved_chunk_texts: List[str],
    relevant_passages: List[RelevantPassage],
    k: int,
) -> float:
    top_k = retrieved_chunk_texts[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(
        1 for chunk in top_k if passages_to_chunk_relevance(chunk, relevant_passages)
    )
    return float(relevant_count) / float(len(top_k))


def mrr(
    retrieved_chunk_texts: List[str], relevant_passages: List[RelevantPassage]
) -> float:
    for rank, chunk in enumerate(retrieved_chunk_texts, start=1):
        if passages_to_chunk_relevance(chunk, relevant_passages):
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(
    retrieved_chunk_texts: List[str],
    relevant_passages: List[RelevantPassage],
    k: int,
) -> float:
    top_k = retrieved_chunk_texts[:k]
    if not top_k or not relevant_passages:
        return 0.0

    dcg = 0.0
    for i, chunk in enumerate(top_k):
        rel = 1.0 if passages_to_chunk_relevance(chunk, relevant_passages) else 0.0
        dcg += rel / np.log2(i + 2)

    ideal_rels = [1.0] * min(len(relevant_passages), k)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def bootstrap_ci(
    values: List[float],
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Returns (point_estimate, lower_ci, upper_ci) deterministically by resampling QUERIES."""
    if not values:
        return (0.0, 0.0, 0.0)

    arr = np.array(values, dtype=float)
    point_estimate = float(np.mean(arr))

    rng = np.random.default_rng(seed)
    boot_means = []
    n_samples = len(arr)

    for _ in range(n):
        sample = rng.choice(arr, size=n_samples, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    return (point_estimate, lower, upper)


def bootstrap_difference_ci(
    adaptive_values: List[float],
    baseline_values: List[float],
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate bootstrapped 95% CI for the difference (adaptive - baseline).

    Resamples QUERIES paired across both arms.
    Returns (diff_point_estimate, lower_ci, upper_ci).
    """
    if not adaptive_values or not baseline_values or len(adaptive_values) != len(baseline_values):
        return (0.0, 0.0, 0.0)

    ad_arr = np.array(adaptive_values, dtype=float)
    base_arr = np.array(baseline_values, dtype=float)
    diff_arr = ad_arr - base_arr
    diff_point_estimate = float(np.mean(diff_arr))

    rng = np.random.default_rng(seed)
    n_samples = len(diff_arr)
    boot_diffs = []

    for _ in range(n):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_diffs.append(np.mean(diff_arr[indices]))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_diffs, alpha * 100))
    upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

    return (diff_point_estimate, lower, upper)


def evaluate_system(
    eval_items: List[EvalItem],
    retriever_fn: Any,
    cfg: Config,
) -> Dict[str, Any]:
    """Evaluate retrieval system across eval dataset items.

    Excludes items with provenance: synthetic_rejected and records excluded count.
    Computes bootstrapped CIs for Recall@5, Recall@10, nDCG@10, MRR, latency, cost.
    """
    seed = int(cfg.raw.get("seed", 42))
    bootstrap_samples = int(cfg.raw.get("eval", {}).get("bootstrap_samples", 1000))
    ci_level = float(cfg.raw.get("eval", {}).get("ci", 0.95))

    valid_items = [item for item in eval_items if item.provenance != "synthetic_rejected"]
    excluded_count = len(eval_items) - len(valid_items)

    recalls_5 = []
    recalls_10 = []
    ndcgs_10 = []
    mrrs = []
    latencies_ms = []

    for item in valid_items:
        candidates, trace = retriever_fn(item.query)
        chunk_texts = [c.text for c in candidates]

        recalls_5.append(recall_at_k(chunk_texts, item.relevant_passages, k=5))
        recalls_10.append(recall_at_k(chunk_texts, item.relevant_passages, k=10))
        ndcgs_10.append(ndcg_at_k(chunk_texts, item.relevant_passages, k=10))
        mrrs.append(mrr(chunk_texts, item.relevant_passages))
        latencies_ms.append(float(trace.get("total_elapsed_ms", 0.0)))

    results = {
        "total_evaluated": len(valid_items),
        "excluded_count": excluded_count,
        "metrics": {
            "recall_5": bootstrap_ci(recalls_5, n=bootstrap_samples, ci=ci_level, seed=seed),
            "recall_10": bootstrap_ci(recalls_10, n=bootstrap_samples, ci=ci_level, seed=seed),
            "ndcg_10": bootstrap_ci(ndcgs_10, n=bootstrap_samples, ci=ci_level, seed=seed),
            "mrr": bootstrap_ci(mrrs, n=bootstrap_samples, ci=ci_level, seed=seed),
            "latency_p50_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
            "latency_p95_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        },
        "per_query_recalls_10": recalls_10,
        "per_query_ndcgs_10": ndcgs_10,
        "per_query_mrrs": mrrs,
        "per_query_latencies": latencies_ms,
    }

    print(
        f"Evaluation Complete | Total Evaluated: {len(valid_items)} | "
        f"Excluded (synthetic_rejected): {excluded_count}"
    )
    return results


def compare_arms(
    eval_items: List[EvalItem],
    baseline_fn: Any,
    adaptive_fn: Any,
    cfg: Config,
) -> Dict[str, Any]:
    """Compare baseline and adaptive systems side by side with CI of the difference."""
    baseline_res = evaluate_system(eval_items, baseline_fn, cfg)
    adaptive_res = evaluate_system(eval_items, adaptive_fn, cfg)

    seed = int(cfg.raw.get("seed", 42))
    bootstrap_samples = int(cfg.raw.get("eval", {}).get("bootstrap_samples", 1000))
    ci_level = float(cfg.raw.get("eval", {}).get("ci", 0.95))

    diff_recall_10 = bootstrap_difference_ci(
        adaptive_res["per_query_recalls_10"],
        baseline_res["per_query_recalls_10"],
        n=bootstrap_samples,
        ci=ci_level,
        seed=seed,
    )
    diff_ndcg_10 = bootstrap_difference_ci(
        adaptive_res["per_query_ndcgs_10"],
        baseline_res["per_query_ndcgs_10"],
        n=bootstrap_samples,
        ci=ci_level,
        seed=seed,
    )
    diff_mrr = bootstrap_difference_ci(
        adaptive_res["per_query_mrrs"],
        baseline_res["per_query_mrrs"],
        n=bootstrap_samples,
        ci=ci_level,
        seed=seed,
    )

    comparison = {
        "total_evaluated": adaptive_res["total_evaluated"],
        "excluded_count": adaptive_res["excluded_count"],
        "baseline": baseline_res["metrics"],
        "adaptive": adaptive_res["metrics"],
        "difference_adaptive_minus_baseline": {
            "recall_10_diff": diff_recall_10,
            "ndcg_10_diff": diff_ndcg_10,
            "mrr_diff": diff_mrr,
        },
    }

    return comparison
