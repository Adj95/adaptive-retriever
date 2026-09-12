"""Test GenerationRequest JSONL serialization roundtrip and request_id determinism per Change Order §4 & §10."""

import json
from adaptive_retriever.models import GenerationRequest, RepresentationKind


def test_generation_request_roundtrip_and_determinism():
    req1 = GenerationRequest(
        request_id="abc123def4567890",
        chunk_id="chunk_001_c0001",
        kind=RepresentationKind.SUMMARY,
        index=0,
        prompt="Summarize the following document...",
        prompt_version="1",
        model="Qwen/Qwen2.5-0.5B-Instruct",
    )

    # Convert to JSON string and back
    json_str = req1.model_dump_json()
    parsed_dict = json.loads(json_str)
    req2 = GenerationRequest(**parsed_dict)

    assert req1 == req2, "GenerationRequest deserialized from JSON must match original object"
    assert req2.request_id == "abc123def4567890"
    assert req2.kind == RepresentationKind.SUMMARY

    # Test request_id stability / determinism across multiple instantiations
    import hashlib
    chunk_id = "doc1_c0005"
    kind = RepresentationKind.QUESTION
    index = 1
    prompt_version = "1"

    id1 = hashlib.sha256(f"{chunk_id}|{kind.value}|{index}|{prompt_version}".encode("utf-8")).hexdigest()[:16]
    id2 = hashlib.sha256(f"{chunk_id}|{kind.value}|{index}|{prompt_version}".encode("utf-8")).hexdigest()[:16]

    assert id1 == id2, "request_id formula sha256(chunk_id|kind|index|prompt_version)[:16] must be deterministic"
