"""Unit test for scripts/demo.py formatting and stub execution."""

from adaptive_retriever.models import RetrievalCandidate
from scripts.demo import format_demo_output, check_collection_exists, get_resolved_device


def test_format_demo_output_contains_all_four_sections():
    query_text = "Do statin drugs cause breast cancer?"
    known_info = ("PLAIN-2", {"MED-10", "MED-14"})
    candidates = [
        RetrievalCandidate(
            chunk_id="MED-10_c0000",
            rerank_score=0.847,
            text="Statins and breast cancer risk study text.",
            metadata={"doc_id": "MED-10"},
        ),
        RetrievalCandidate(
            chunk_id="MED-14_c0000",
            rerank_score=0.832,
            text="Clinical survival rates for statin patients.",
            metadata={"doc_id": "MED-14"},
        ),
    ]
    trace = {
        "stages": {
            "rerank": {
                "decision": "reranked",
                "gate": {"margin": 0.11, "threshold": 0.30},
            }
        }
    }
    answer_text = "Statins do not increase breast cancer risk according to clinical studies. Overall mortality was reduced."
    attributions = [
        ("Statins do not increase breast cancer risk according to clinical studies.", "MED-10_c0000"),
        ("Overall mortality was reduced.", None),
    ]
    timings = {
        "retrieval_ms": 118.0,
        "rerank_ms": 84.0,
        "generation_ms": 1290.0,
        "total_ms": 1492.0,
    }

    out = format_demo_output(
        query_text=query_text,
        known_info=known_info,
        candidates=candidates,
        trace=trace,
        answer_text=answer_text,
        attributions=attributions,
        timings=timings,
    )

    # Verify all 4 major sections are present in output
    assert "QUERY     Do statin drugs cause breast cancer?" in out
    assert "NFCorpus test query PLAIN-2, 2 gold documents" in out
    assert "RETRIEVED top 2 of 2" in out
    assert "MED-10_c0000" in out
    assert "gold: yes" in out
    assert "rerank fired: yes (top-1 margin 0.11 < 0.30)" in out
    assert "ANSWER" in out
    assert "Statins do not increase breast cancer risk" in out
    assert "GROUNDING 1 of 2 sentences matched >= 8-gram verbatim to retrieved context" in out
    assert "sentence 1  grounded   -> MED-10_c0000" in out
    assert "sentence 2  ungrounded" in out
    assert "COST      retrieval 118 ms | rerank 84 ms | generation 1,290 ms | total 1,492 ms" in out


def test_resolved_device_string():
    dev = get_resolved_device()
    assert isinstance(dev, str)
    assert len(dev) > 0
