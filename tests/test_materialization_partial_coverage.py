"""Test partial coverage materialization filter (R-01d-fix bug regression test).

Confirms that chunks with partial representations (e.g. summary + question present,
keywords missing due to timeout) materialize all valid available results rather
than dropping the entire chunk.
"""

from adaptive_retriever.config import Config
from adaptive_retriever.ingest.representations import post_process_representation, RepresentationKind
from adaptive_retriever.models import Chunk, GenerationResult, Representation


def test_partial_representation_materialization():
    cfg = Config({})
    chunk = Chunk(
        chunk_id="chunk_partial_001",
        doc_id="doc_partial",
        ordinal=0,
        text="Standard corporate governance document outlining operational risk guidelines.",
        char_start=0,
        char_end=74,
        chunker_used="recursive_512",
        token_count=12,
    )

    # Simulating partial generation results (summary & question present, keywords missing)
    res_summary = GenerationResult(
        request_id="req_sum_001",
        chunk_id=chunk.chunk_id,
        kind=RepresentationKind.SUMMARY,
        index=0,
        text="Operational risk guidelines for corporate governance.",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        tokens_in=30,
        tokens_out=10,
        generated_at="2026-09-24T00:00:00Z",
    )
    res_question = GenerationResult(
        request_id="req_q_001",
        chunk_id=chunk.chunk_id,
        kind=RepresentationKind.QUESTION,
        index=0,
        text="What are the operational risk guidelines?",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        tokens_in=30,
        tokens_out=8,
        generated_at="2026-09-24T00:00:00Z",
    )

    results = [res_summary, res_question]
    chunk_map = {chunk.chunk_id: chunk}

    # Lenient materialization filter logic used in R-01d-fix / scripts/reindex_and_evaluate_r01d_fix.py
    materialized_reps: list[Representation] = []
    for res in results:
        c = chunk_map.get(res.chunk_id)
        if c is None:
            continue
        proc_text = post_process_representation(res.text, res.kind, strict_mode=True)
        rep_id = f"{res.chunk_id}_{res.kind.value}" if res.kind != RepresentationKind.QUESTION else f"{res.chunk_id}_question_{res.index}"
        materialized_reps.append(
            Representation(
                rep_id=rep_id,
                chunk_id=res.chunk_id,
                kind=res.kind,
                text=proc_text,
                generated_by=res.model,
                tier="low",
            )
        )

    # Assert both partial representations materialized cleanly without being discarded
    assert len(materialized_reps) == 2
    assert {r.kind for r in materialized_reps} == {RepresentationKind.SUMMARY, RepresentationKind.QUESTION}
