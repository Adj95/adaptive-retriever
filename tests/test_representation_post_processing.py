"""Test for representation post-processing and preamble stripping."""

import pytest
import re
from adaptive_retriever.models import RepresentationKind
from adaptive_retriever.ingest.representations import (
    post_process_representation,
    PREAMBLE_PATTERNS,
)


def test_post_process_strips_preamble_summary():
    raw = "Here is a summary of the provided text: The system processes incoming documents using adaptive retrieval."
    cleaned = post_process_representation(raw, RepresentationKind.SUMMARY)
    assert not any(pat.match(cleaned) for pat in PREAMBLE_PATTERNS)
    assert cleaned == "The system processes incoming documents using adaptive retrieval."


def test_post_process_strips_preamble_question():
    raw = "Based on the provided text, what is the throughput of the ingestion pipeline?"
    cleaned = post_process_representation(raw, RepresentationKind.QUESTION)
    assert not any(pat.match(cleaned) for pat in PREAMBLE_PATTERNS)
    assert cleaned == "What is the throughput of the ingestion pipeline?"


def test_post_process_keywords_format():
    raw = "Here are some potential keywords related to the given text:\n1. retrieval\n2. embeddings\n3. vector database"
    cleaned = post_process_representation(raw, RepresentationKind.KEYWORDS)
    assert "\n" not in cleaned
    assert not re.search(r"\d+\.\s", cleaned)
    assert not any(pat.match(cleaned) for pat in PREAMBLE_PATTERNS)
    assert cleaned == "retrieval, embeddings, vector database"


def test_post_process_raises_on_empty_or_pure_preamble():
    raw = "Here is a summary of the provided text:"
    with pytest.raises(ValueError):
        post_process_representation(raw, RepresentationKind.SUMMARY)
