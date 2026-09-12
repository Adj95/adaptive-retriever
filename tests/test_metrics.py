"""Hand-computed fixture unit tests for IR metrics and relevance functions (spec §8.3 & Addendum B4)."""

import pytest
from adaptive_retriever.eval.metrics import (
    normalise,
    is_relevant,
    passages_to_chunk_relevance,
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    bootstrap_ci,
)
from adaptive_retriever.models import RelevantPassage


def test_normalise():
    raw_text = "  Hello,   WORLD! This is a test passage with 12.5% accuracy and pre-processing.  "
    norm = normalise(raw_text)
    assert norm == "hello world this is a test passage with 12.5 accuracy and pre-processing"


def test_is_relevant_both_conditions():
    passage_text = (
        "The adaptive retriever system evaluates query complexity and routes queries "
        "to the optimal retrieval cascade based on detected document features."
    )
    passage_norm = normalise(passage_text)
    passage = RelevantPassage(
        doc_id="doc1",
        passage_text=passage_text,
        passage_norm=passage_norm,
        canonical_char_start=0,
        canonical_char_end=len(passage_text),
        canonical_parser="pymupdf",
    )

    # 1. Matching chunk: contains tokens and exact 8-gram
    matching_chunk = (
        "In our research, the adaptive retriever system evaluates query complexity and routes queries "
        "to the optimal retrieval cascade based on detected document features efficiently."
    )
    assert is_relevant(matching_chunk, passage) is True

    # 2. Deliberate near-miss: contains high token overlap (>60%) BUT NO contiguous 8-gram
    near_miss_chunk = (
        "evaluates adaptive retriever system query complexity routes and to optimal "
        "the retrieval cascade based detected document features on."
    )
    # Token overlap is high, but 8-grams are shuffled
    assert is_relevant(near_miss_chunk, passage) is False


def test_ir_metrics_fixtures():
    p1_text = "The quick brown fox jumps over the lazy dog in the sunny meadow today."
    p1 = RelevantPassage(
        doc_id="doc1",
        passage_text=p1_text,
        passage_norm=normalise(p1_text),
        canonical_char_start=0,
        canonical_char_end=len(p1_text),
        canonical_parser="pymupdf",
    )
    relevant_passages = [p1]

    # Rank 1: irrelevant chunk, Rank 2: relevant chunk
    chunk_irrelevant = "Sphinx of black quartz, judge my vow."
    chunk_relevant = p1_text

    retrieved = [chunk_irrelevant, chunk_relevant]

    assert recall_at_k(retrieved, relevant_passages, k=1) == 0.0
    assert recall_at_k(retrieved, relevant_passages, k=2) == 1.0

    assert precision_at_k(retrieved, relevant_passages, k=1) == 0.0
    assert precision_at_k(retrieved, relevant_passages, k=2) == 0.5

    assert mrr(retrieved, relevant_passages) == 0.5  # First relevant item at rank 2 -> 1/2 = 0.5

    # Hand-computed nDCG@2: rel_1 = 0, rel_2 = 1.0. DCG = 1.0 / log2(3) = 1.0 / 1.58496 = 0.6309. IDCG = 1.0 / log2(2) = 1.0
    ndcg = ndcg_at_k(retrieved, relevant_passages, k=2)
    assert pytest.approx(ndcg, abs=1e-3) == 0.6309


def test_bootstrap_ci_determinism():
    values = [0.8, 0.9, 0.85, 0.7, 0.95, 0.88, 0.92]
    res1 = bootstrap_ci(values, seed=42)
    res2 = bootstrap_ci(values, seed=42)
    assert res1 == res2
    assert res1[0] == pytest.approx(sum(values) / len(values))
    assert res1[1] <= res1[0] <= res1[2]
