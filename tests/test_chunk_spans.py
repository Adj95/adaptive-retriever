"""Unit tests verifying chunk character span exactness for all 5 spec chunkers."""

from adaptive_retriever.ingest.chunkers import chunk_document
from adaptive_retriever.ingest.parsers import ParsedDocument


def test_chunk_spans_recursive_512():
    text = "Line 1: Hello World.\nLine 2: Section body content text.\nLine 3: More information.\n" * 30
    parsed = ParsedDocument(text=text)
    chunks = chunk_document(parsed, doc_id="test_doc_01", chunker_name="recursive_512")

    assert len(chunks) > 0
    for c in chunks:
        assert parsed.text[c.char_start:c.char_end] == c.text


def test_chunk_spans_structural():
    text = "# Section 1\nThis is section one text.\n\n# Section 2\nThis is section two text.\n"
    parsed = ParsedDocument(text=text)
    chunks = chunk_document(parsed, doc_id="test_doc_02", chunker_name="structural")

    assert len(chunks) > 0
    for c in chunks:
        assert parsed.text[c.char_start:c.char_end] == c.text


def test_chunk_spans_syntax_aware():
    text = "import os\n\ndef func_one():\n    return 42\n\ndef func_two():\n    return 'hello'\n"
    parsed = ParsedDocument(text=text)
    chunks = chunk_document(parsed, doc_id="test_doc_03", chunker_name="syntax_aware")

    assert len(chunks) > 0
    for c in chunks:
        assert parsed.text[c.char_start:c.char_end] == c.text


def test_chunk_spans_table_preserving():
    text = "Intro text.\n\n| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |\n\nConclusion text.\n"
    parsed = ParsedDocument(text=text)
    chunks = chunk_document(parsed, doc_id="test_doc_04", chunker_name="table_preserving")

    assert len(chunks) > 0
    for c in chunks:
        assert parsed.text[c.char_start:c.char_end] == c.text


def test_chunk_spans_speaker_turn():
    text = "CHAIRMAN: Welcome to the meeting.\nMR. SMITH: Thank you Mr. Chairman.\nMS. JONES: I agree.\n"
    parsed = ParsedDocument(text=text)
    chunks = chunk_document(parsed, doc_id="test_doc_05", chunker_name="speaker_turn")

    assert len(chunks) > 0
    for c in chunks:
        assert parsed.text[c.char_start:c.char_end] == c.text
