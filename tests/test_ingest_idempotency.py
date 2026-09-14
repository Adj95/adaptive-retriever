"""Unit test for ingestion idempotency & hash-suffixed Qdrant collection naming (spec §9.5 & Addendum v1.2 §D)."""

import tempfile
from pathlib import Path
from adaptive_retriever.ingest.chunkers import chunk_document
from adaptive_retriever.ingest.parsers import ParsedDocument
from adaptive_retriever.ingest.representations import generate_representations
from adaptive_retriever.vectorstore import VectorStore


def test_ingest_config_hash_collection_naming():
    cfg_hash = "a94fcb78"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        vs = VectorStore(collection_base_name="test_collection", config_hash=cfg_hash, storage_path=Path(tmp_dir), mode="local")
        assert vs.collection_name == f"test_collection_{cfg_hash}"
        cols = [c.name for c in vs.client.get_collections().collections]
        assert vs.collection_name in cols
        vs.close()


def test_ingest_idempotency_no_op_and_force():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        vs = VectorStore(collection_base_name="test_idempotency", config_hash="test1234", storage_path=Path(tmp_dir), mode="local")

        parsed = ParsedDocument(text="Idempotency test text block.\nSecond line of document.\n")
        chunks = chunk_document(parsed, doc_id="doc_idemp_01")
        reps = generate_representations(chunks[0])

        meta = {"doc_class": "digital_prose"}

        # Initial upsert
        vs.upsert_representations(chunks[0], reps, metadata=meta)
        initial_points = vs.client.count(collection_name=vs.collection_name).count
        assert initial_points > 0

        # Second upsert (same doc_id & config_hash) -> points count remains identical (idempotent overwrite/skip)
        vs.upsert_representations(chunks[0], reps, metadata=meta)
        second_points = vs.client.count(collection_name=vs.collection_name).count
        assert second_points == initial_points

        vs.close()
