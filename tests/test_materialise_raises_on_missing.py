"""Test that materialise_representations raises a ValueError on missing request_id per Change Order §5 & §10."""

import pytest
from adaptive_retriever.config import Config
from adaptive_retriever.ingest.representations import materialise_representations, plan_representations
from adaptive_retriever.models import Chunk, GenerationResult, RepresentationKind


def test_materialise_raises_on_missing_request_id():
    cfg = Config({})
    chunk_text = (
        "Dense financial analysis report containing balance sheet figures and revenue streams. "
        "Key metrics include operating margins and asset allocations across diverse global investment portfolios. "
        "Enterprise value calculations rely heavily on discounted cash flow projections, EBITDA multiples, "
        "working capital adjustments, debt obligations, interest rate sensitivities, regulatory compliance frameworks, "
        "shareholder equity distributions, capital expenditures, and audited financial statements from external accounting entities."
    )
    chunk = Chunk(
        chunk_id="chunk_test_001",
        doc_id="doc_test",
        ordinal=0,
        text=chunk_text,
        char_start=0,
        char_end=len(chunk_text),
        chunker_used="recursive_512",
        token_count=60,
    )

    det_reps, requests = plan_representations([chunk], cfg)
    assert len(requests) > 0, "High-density chunk should produce generation requests"

    # Pass empty results list
    empty_results = []
    with pytest.raises((ValueError, RuntimeError), match="Missing generation result for request_id"):
        materialise_representations([chunk], det_reps, empty_results, cfg)

    # Pass results list missing one request_id
    incomplete_result = GenerationResult(
        request_id="non_matching_request_id",
        chunk_id=chunk.chunk_id,
        kind=RepresentationKind.SUMMARY,
        index=0,
        text="A valid generated summary.",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        tokens_in=10,
        tokens_out=15,
        generated_at="2026-09-09T00:00:00Z",
    )
    with pytest.raises((ValueError, RuntimeError), match="Missing generation result for request_id"):
        materialise_representations([chunk], det_reps, [incomplete_result], cfg)
