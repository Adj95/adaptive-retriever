"""Unit test verifying invariant §6.2: RepresentationKind.RAW returns chunk.text exactly."""

from adaptive_retriever.ingest.representations import generate_representations
from adaptive_retriever.models import Chunk, RepresentationKind


def test_representation_raw_invariant():
    chunk_text = "This is exact chunk text that MUST be preserved invariant without mutation or summary."
    chunk = Chunk(
        chunk_id="chunk_test_001",
        doc_id="doc_test_001",
        ordinal=0,
        text=chunk_text,
        char_start=0,
        char_end=len(chunk_text),
        chunker_used="recursive_512",
        token_count=13,
    )

    reps = generate_representations(chunk, doc_title="Test Doc")

    raw_reps = [r for r in reps if r.kind == RepresentationKind.RAW]
    assert len(raw_reps) == 1
    assert raw_reps[0].text == chunk_text
    assert raw_reps[0].text == chunk.text
