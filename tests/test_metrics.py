"""Hand-computed fixture unit tests for IR metrics and relevance functions (spec §8.3 & Addendum B4)."""

import pytest
from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import (
    normalise,
    is_relevant,
    passages_to_chunk_relevance,
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    bootstrap_ci,
    bootstrap_difference_ci,
    evaluate_system,
)
from adaptive_retriever.models import DocClass, EvalItem, RelevantPassage, RetrievalCandidate


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

    matching_chunk = (
        "In our research, the adaptive retriever system evaluates query complexity and routes queries "
        "to the optimal retrieval cascade based on detected document features efficiently."
    )
    assert is_relevant(matching_chunk, passage) is True

    near_miss_chunk = (
        "evaluates adaptive retriever system query complexity routes and to optimal "
        "the retrieval cascade based detected document features on."
    )
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

    chunk_irrelevant = "Sphinx of black quartz, judge my vow."
    chunk_relevant = p1_text

    retrieved = [chunk_irrelevant, chunk_relevant]

    assert recall_at_k(retrieved, relevant_passages, k=1) == 0.0
    assert recall_at_k(retrieved, relevant_passages, k=2) == 1.0

    assert precision_at_k(retrieved, relevant_passages, k=1) == 0.0
    assert precision_at_k(retrieved, relevant_passages, k=2) == 0.5

    assert mrr(retrieved, relevant_passages) == 0.5

    ndcg = ndcg_at_k(retrieved, relevant_passages, k=2)
    assert pytest.approx(ndcg, abs=1e-3) == 0.6309


def test_bootstrap_ci_determinism_and_sample_size():
    import numpy as np
    values = [0.8, 0.9, 0.85, 0.7, 0.95, 0.88, 0.92, 0.75, 0.6, 0.99] * 10

    # Determinism check
    assert bootstrap_ci(values, n=1000, seed=42) == bootstrap_ci(values, n=1000, seed=42)

    # 1000 samples has lower variance across seeds than 50 samples
    bounds_1000 = [bootstrap_ci(values, n=1000, seed=s)[1] for s in range(10)]
    bounds_50 = [bootstrap_ci(values, n=50, seed=s)[1] for s in range(10)]
    assert np.var(bounds_1000) < np.var(bounds_50)


def test_excluded_synthetic_rejected_changes_denominator(capsys):
    p_text = "Sample passage text for evaluation testing."
    p = RelevantPassage(
        doc_id="doc1",
        passage_text=p_text,
        passage_norm=normalise(p_text),
        canonical_char_start=0,
        canonical_char_end=len(p_text),
        canonical_parser="pymupdf",
    )

    item_valid = EvalItem(
        item_id="item_1",
        query="query 1",
        relevant_passages=[p],
        doc_class_of_source=DocClass.DIGITAL_PROSE,
        provenance="human",
        split="dev",
    )
    item_rejected = EvalItem(
        item_id="item_2",
        query="query 2",
        relevant_passages=[p],
        doc_class_of_source=DocClass.DIGITAL_PROSE,
        provenance="synthetic_rejected",
        split="dev",
    )

    eval_items = [item_valid, item_rejected]

    def mock_retriever(query):
        return [RetrievalCandidate(chunk_id="c1", text=p_text, final_rank=1)], {"total_elapsed_ms": 10.0}

    cfg = Config.load_default() if hasattr(Config, "load_default") else Config.load("config/default.yaml")
    results = evaluate_system(eval_items, mock_retriever, cfg)

    assert results["total_evaluated"] == 1
    assert results["excluded_count"] == 1

    captured = capsys.readouterr()
    assert "Excluded (synthetic_rejected): 1" in captured.out


def test_ndcg_bounded_when_multiple_matching_chunks():
    """Verify nDCG <= 1.0 when 9 retrieved chunks match a single gold passage."""
    p_text = "The quick brown fox jumps over the lazy dog in the sunny meadow today."
    p = RelevantPassage(
        doc_id="doc1",
        passage_text=p_text,
        passage_norm=normalise(p_text),
        canonical_char_start=0,
        canonical_char_end=len(p_text),
        canonical_parser="pymupdf",
    )
    # 9 retrieved chunks all match 1 gold passage
    retrieved = [p_text] * 9
    ndcg = ndcg_at_k(retrieved, [p], k=10)
    assert 0.0 <= ndcg <= 1.0, f"nDCG must be <= 1.0, got {ndcg}"
    assert pytest.approx(ndcg, abs=1e-6) == 1.0


def test_ndcg_bounded_three_matching_chunks():
    """Verify nDCG <= 1.0 when a gold passage matches 3 chunks."""
    p_text = "The quick brown fox jumps over the lazy dog in the sunny meadow today."
    p = RelevantPassage(
        doc_id="doc1",
        passage_text=p_text,
        passage_norm=normalise(p_text),
        canonical_char_start=0,
        canonical_char_end=len(p_text),
        canonical_parser="pymupdf",
    )
    retrieved = [p_text, p_text, p_text]
    ndcg = ndcg_at_k(retrieved, [p], k=5)
    assert 0.0 <= ndcg <= 1.0, f"nDCG must be <= 1.0, got {ndcg}"
    assert pytest.approx(ndcg, abs=1e-6) == 1.0


def test_recall_and_mrr_bounds_and_assertions():
    """Verify recall_at_k and mrr are bounded in [0, 1]."""
    p_text = "The quick brown fox jumps over the lazy dog in the sunny meadow today."
    p = RelevantPassage(
        doc_id="doc1",
        passage_text=p_text,
        passage_norm=normalise(p_text),
        canonical_char_start=0,
        canonical_char_end=len(p_text),
        canonical_parser="pymupdf",
    )
    retrieved = [p_text] * 5
    r = recall_at_k(retrieved, [p], k=5)
    m = mrr(retrieved, [p])
    assert 0.0 <= r <= 1.0
    assert 0.0 <= m <= 1.0

