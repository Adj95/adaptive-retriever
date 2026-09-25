import pytest
from adaptive_retriever.benchmarks.beir import is_relevant_beir, get_qrel_score_beir

class DummyChunk:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id

def test_beir_relevance_contract():
    qrels = {
        "q1": {
            "doc_gold_high": 2,
            "doc_gold_low": 1,
            "doc_irrelevant": 0,
        }
    }
    
    chunk_high = DummyChunk("doc_gold_high")
    chunk_low = DummyChunk("doc_gold_low")
    chunk_zero = DummyChunk("doc_irrelevant")
    chunk_absent = DummyChunk("doc_absent")

    # Score 2 is relevant
    assert is_relevant_beir(chunk_high, "q1", qrels) is True
    assert get_qrel_score_beir(chunk_high, "q1", qrels) == 2

    # Score 1 is relevant
    assert is_relevant_beir(chunk_low, "q1", qrels) is True
    assert get_qrel_score_beir(chunk_low, "q1", qrels) == 1

    # Score 0 is NOT relevant
    assert is_relevant_beir(chunk_zero, "q1", qrels) is False
    assert get_qrel_score_beir(chunk_zero, "q1", qrels) == 0

    # Absent doc is NOT relevant
    assert is_relevant_beir(chunk_absent, "q1", qrels) is False
    assert get_qrel_score_beir(chunk_absent, "q1", qrels) == 0

    # Query absent from qrels is NOT relevant
    assert is_relevant_beir(chunk_high, "q99", qrels) is False
    assert get_qrel_score_beir(chunk_high, "q99", qrels) == 0
