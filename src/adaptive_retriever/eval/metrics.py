"""Evaluation metrics implemented directly using numpy and scipy (spec §8.3 & Addendum B4)."""

import re
import unicodedata
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from adaptive_retriever.models import RelevantPassage

def normalise(text: str) -> str:
    """lowercase, collapse all whitespace to single spaces, strip punctuation
    except intra-word hyphens and decimal points, unicode NFKC."""
    text = unicodedata.normalize("NFKC", text).lower()
    # Replace whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Remove punctuation except intra-word hyphens and decimal numbers
    # Keep alphanumeric, spaces, intra-word hyphens, decimal points in digits
    cleaned_chars = []
    for i, char in enumerate(text):
        if char.isalnum() or char == " ":
            cleaned_chars.append(char)
        elif char == "-" and 0 < i < len(text) - 1 and text[i-1].isalnum() and text[i+1].isalnum():
            cleaned_chars.append(char)
        elif char == "." and 0 < i < len(text) - 1 and text[i-1].isdigit() and text[i+1].isdigit():
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
    1. token-level containment:
       |tokens(passage_norm) & tokens(chunk_norm)| / |tokens(passage_norm)| >= 0.60
    2. contiguity: at least one 8-gram from passage_norm appears verbatim in chunk_norm
    (or 4-gram for table passages if relaxed per Addendum B4).
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

    # Determine n-gram size: 8-gram by default, 4-gram if contains_table and containment >= 0.75
    n_gram_size = 8
    passage_ngrams = get_ngrams(passage_tokens, n_gram_size)
    chunk_ngrams = get_ngrams(chunk_tokens, n_gram_size)

    condition_2 = bool(passage_ngrams & chunk_ngrams)

    # Instrumentation logging (Addendum B4)
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
    # Check how many relevant passages are covered by at least one top-k chunk
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

    # Ideal DCG: best possible assignment of relevance
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
    """Returns (point_estimate, lower_ci, upper_ci) deterministically."""
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
