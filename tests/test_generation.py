"""Unit tests for grounded generation with citations (Task C, spec §6)."""

from unittest.mock import MagicMock
import pytest

from adaptive_retriever.config import Config
from adaptive_retriever.generation.generate import GeneratedAnswer, generate_answer, parse_citations_defensive
from adaptive_retriever.models import RetrievalCandidate


def make_mock_candidate(chunk_id: str, text: str = "Sample passage text.") -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        text=text,
        score=0.9,
        dense_rank=1,
        sparse_rank=1,
        source="hybrid",
    )


class MockLLMClient:
    def __init__(self, response_text: str = "The method is testing [chunk_1]."):
        self.response_text = response_text
        self.call_count = 0
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        return self.response_text


def test_generate_answer_resolvable_citation(tmp_path, monkeypatch):
    monkeypatch.setattr("adaptive_retriever.cache.CACHE_DIR", tmp_path / "cache")
    cands = [make_mock_candidate("chunk_1", "Testing method.")]
    llm = MockLLMClient("The method is testing [chunk_1].")
    cfg = MagicMock()
    cfg.raw = {"generation": {"enabled": True, "cite_chunks": True, "citation_mode": "model"}}
    
    ans = generate_answer("What method?", cands, cfg=cfg, llm=llm)
    
    assert isinstance(ans, GeneratedAnswer)
    assert not ans.refused
    assert ans.citations == ["chunk_1"]
    assert "testing" in ans.text


def test_hallucinated_citation_raises(tmp_path, monkeypatch):
    """Rule §6: A citation to a chunk that was not in context is a hallucinated citation — raise ValueError."""
    monkeypatch.setattr("adaptive_retriever.cache.CACHE_DIR", tmp_path / "cache")
    cands = [make_mock_candidate("chunk_1", "Real passage.")]
    llm = MockLLMClient("Here is an answer citing [chunk_999].")
    cfg = MagicMock()
    cfg.raw = {"generation": {"enabled": True, "cite_chunks": True, "citation_mode": "model", "strict_citations": True}}
    
    with pytest.raises(ValueError, match="Hallucinated citation 'chunk_999'"):
        generate_answer("What other method?", cands, cfg=cfg, llm=llm)


def test_generation_cache_roundtrip(tmp_path, monkeypatch):
    """Cache on (query, ordered chunk_ids, prompt_version, model)."""
    monkeypatch.setattr("adaptive_retriever.cache.CACHE_DIR", tmp_path / "cache")
    
    cands = [make_mock_candidate("chunk_10")]
    llm = MockLLMClient("Cached answer with [chunk_10].")
    cfg = MagicMock()
    cfg.raw = {"generation": {"enabled": True, "cite_chunks": True, "citation_mode": "model"}}
    
    ans1 = generate_answer("Cache query?", cands, cfg=cfg, llm=llm)
    assert llm.call_count == 1
    
    ans2 = generate_answer("Cache query?", cands, cfg=cfg, llm=llm)
    assert llm.call_count == 1  # Served from cache
    assert ans2.text == ans1.text
    assert ans2.citations == ans1.citations


def test_refuse_if_confidence_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr("adaptive_retriever.cache.CACHE_DIR", tmp_path / "cache")
    from pathlib import Path
    cands = [make_mock_candidate("chunk_1")]
    llm = MockLLMClient("Some answer [chunk_1].")
    cfg = Config.load(Path("config/default.yaml"))
    
    ans = generate_answer("Low confidence query?", cands, cfg=cfg, llm=llm, confidence=0.05)
    
    assert ans.refused
    assert "below refusal threshold" in ans.refusal_reason
    assert llm.call_count == 0


def test_refuse_on_empty_candidates():
    llm = MockLLMClient("Answer.")
    
    ans = generate_answer("No cands?", [], llm=llm)
    
    assert ans.refused
    assert "No candidate passages" in ans.refusal_reason


def test_zero_citations_refusal_when_cite_chunks_enabled(tmp_path, monkeypatch):
    """When cite_chunks is True, an answer containing no resolvable citations must be flagged as refused with citation_parse_status 'no_citations'."""
    monkeypatch.setattr("adaptive_retriever.cache.CACHE_DIR", tmp_path / "cache")
    cands = [make_mock_candidate("chunk_1", "Real passage text.")]
    llm = MockLLMClient("This is an ungrounded answer without any citation markers.")
    cfg = MagicMock()
    cfg.raw = {"generation": {"enabled": True, "cite_chunks": True, "citation_mode": "model"}}
    
    ans = generate_answer("What ungrounded method?", cands, cfg=cfg, llm=llm)
    
    assert ans.refused is True
    assert ans.refusal_reason == "Answer contained no resolvable citations."
    assert ans.trace.get("citation_parse_status") == "no_citations"


def test_attribution_citation_mode(tmp_path, monkeypatch):
    """Test post-hoc n-gram attribution citation mode (n >= 8 words)."""
    monkeypatch.setattr("adaptive_retriever.cache.CACHE_DIR", tmp_path / "cache")
    passage = "The quick brown fox jumps over the lazy dog in the sunny forest near the river."
    cands = [make_mock_candidate("chunk_fox", passage)]
    llm = MockLLMClient("The quick brown fox jumps over the lazy dog in the sunny forest near the river.")
    cfg = MagicMock()
    cfg.raw = {"generation": {"enabled": True, "cite_chunks": True, "citation_mode": "attribution"}}

    ans = generate_answer("Fox query?", cands, cfg=cfg, llm=llm)
    assert ans.refused is False
    assert ans.citations == ["chunk_fox"]
    assert ans.trace.get("citation_parse_status") == "derived_attribution"
