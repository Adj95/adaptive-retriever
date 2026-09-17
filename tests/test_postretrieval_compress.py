"""Tests for post-retrieval candidate text compression (Task A)."""

import pytest
from typing import List
from adaptive_retriever.models import RetrievalCandidate
from adaptive_retriever.postretrieval.compress import compress


class DummyEmbedClient:
    def encode(self, text: str) -> List[float]:
        # Return deterministic mock embeddings based on keyword match
        if "query" in text.lower() or "python" in text.lower():
            return [1.0, 0.0, 0.0]
        elif "finance" in text.lower():
            return [0.0, 1.0, 0.0]
        else:
            return [0.0, 0.0, 1.0]


class DummyLLM:
    def __init__(self, response: str = "Python is a language."):
        self.response = response

    def complete(self, prompt: str, **kwargs) -> str:
        return self.response


def test_compress_mode_none_passthrough():
    cfg = {"postretrieval": {"compression": {"mode": "none"}}}
    candidates = [
        RetrievalCandidate(chunk_id="c1", text="Sample chunk text for testing.")
    ]
    res, trace = compress("python", candidates, cfg=cfg)
    assert len(res) == 1
    assert res[0].text == "Sample chunk text for testing."
    assert trace["mode"] == "none"
    assert trace["overall_compression_ratio"] == 1.0


def test_compress_verbatim_substring_assertion():
    cfg = {"postretrieval": {"compression": {"mode": "embeddings", "embeddings_threshold": 0.3}}}
    candidates = [
        RetrievalCandidate(
            chunk_id="c1",
            text="Python is a programming language. Financial markets use options and futures. Python is widely used in AI."
        )
    ]
    embedder = DummyEmbedClient()
    res, trace = compress("python programming", candidates, cfg=cfg, embedding_client=embedder)

    assert len(res) == 1
    compressed_text = res[0].text
    # Every sentence span in compressed_text MUST be a verbatim substring of original text
    for line in compressed_text.splitlines():
        if line.strip():
            assert line.strip() in candidates[0].text, f"Line '{line}' is not a verbatim substring of original text!"


def test_compress_paraphrased_text_fails():
    """Asserts that compress() rejects non-verbatim / paraphrased lines from LLM response."""
    cfg = {
        "postretrieval": {
            "compression": {
                "mode": "cascade",
                "embeddings_threshold": 0.35,
                "llm_extractor_if_below": 0.50,
            }
        }
    }
    candidates = [
        RetrievalCandidate(
            chunk_id="c1",
            text="Financial markets use options and futures. Equity markets trade stocks."
        )
    ]
    embedder = DummyEmbedClient()
    # LLM returns one verbatim span and one paraphrased non-verbatim span
    llm = DummyLLM("Equity markets trade stocks.\nFinancial options are great for trading.")

    res, trace = compress("python", candidates, cfg=cfg, llm=llm, embedding_client=embedder)
    assert len(res) == 1
    # Paraphrased span 'Financial options are great for trading.' must be rejected
    assert res[0].text == "Equity markets trade stocks."
    assert "Financial options are great for trading." not in res[0].text


def test_compress_cascade_embeddings_and_llm_tier():
    # Embedding tier high sim -> embeddings tier fires
    cfg_high = {
        "postretrieval": {
            "compression": {
                "mode": "cascade",
                "embeddings_threshold": 0.35,
                "llm_extractor_if_below": 0.50,
            }
        }
    }
    candidates = [
        RetrievalCandidate(
            chunk_id="c1",
            text="Python is a programming language. Financial markets use options and futures."
        )
    ]
    embedder = DummyEmbedClient()
    llm = DummyLLM("Python is a programming language.")

    res_high, trace_high = compress("python", candidates, cfg=cfg_high, llm=llm, embedding_client=embedder)
    assert trace_high["tier_counts"]["embeddings"] == 1
    assert trace_high["tier_counts"]["llm_extractor"] == 0

    # Low similarity -> falls back to LLM extractor tier
    candidates_low = [
        RetrievalCandidate(
            chunk_id="c2",
            text="Financial markets use options and futures. Equity markets trade stocks."
        )
    ]
    res_low, trace_low = compress("python", candidates_low, cfg=cfg_high, llm=llm, embedding_client=embedder)
    assert trace_low["tier_counts"]["llm_extractor"] == 1


def test_compress_ratio_reporting():
    cfg = {"postretrieval": {"compression": {"mode": "embeddings", "embeddings_threshold": 0.3}}}
    candidates = [
        RetrievalCandidate(
            chunk_id="c1",
            text="Python is a programming language. Financial markets use options and futures."
        )
    ]
    embedder = DummyEmbedClient()
    res, trace = compress("python", candidates, cfg=cfg, embedding_client=embedder)

    assert "overall_compression_ratio" in trace
    assert 0.0 <= trace["overall_compression_ratio"] <= 1.0
    cand_meta = res[0].metadata.get("compression", {})
    assert "char_in" in cand_meta
    assert "char_out" in cand_meta
    assert "compression_ratio" in cand_meta
