"""Smoke test for ingestion pipeline using tmp_path and stub LLM client."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from adaptive_retriever.llm import LLMResponse
from adaptive_retriever.ingest.pipeline import run_ingestion_pipeline


class StubLLMClient:
    def __init__(self, *args, **kwargs):
        self.stats = {"truncated_count": {"summary": 0, "question": 0, "keywords": 0}}

    def complete_batch(self, prompts, stage="ingest_representation"):
        return [
            LLMResponse(
                text="This is a stub representation response for testing pipeline smoke.",
                model="stub-model",
                tokens_in=10,
                tokens_out=15,
                usd_cost=0.0,
                cached=False,
            )
            for _ in prompts
        ]


def test_ingest_pipeline_smoke(tmp_path, monkeypatch):
    # 1. Create 3 tiny fixture documents
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("Python programming is powerful and dynamic. " * 15, encoding="utf-8")

    doc2 = tmp_path / "doc2.html"
    doc2.write_text("<html><body><h1>HTML Title</h1><p>" + ("Web development content " * 15) + "</p></body></html>", encoding="utf-8")

    doc3 = tmp_path / "doc3.py"
    doc3.write_text("def hello_world():\n    print('Hello World')\n" * 10, encoding="utf-8")

    # 2. Write manifest in tmp_path
    manifest_p = tmp_path / "MANIFEST.jsonl"
    records = [
        {"doc_path": str(doc1), "expected_doc_class": "digital_prose", "doc_id": "doc1hash12345678"},
        {"doc_path": str(doc2), "expected_doc_class": "html", "doc_id": "doc2hash12345678"},
        {"doc_path": str(doc3), "expected_doc_class": "code", "doc_id": "doc3hash12345678"},
    ]
    manifest_p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    # 3. Patch STATS_REPORT_PATH, PARSE_CACHE_DIR, and LLM client
    report_p = tmp_path / "ingest_stats.json"
    monkeypatch.setattr("adaptive_retriever.ingest.pipeline.STATS_REPORT_PATH", report_p)
    monkeypatch.setattr("adaptive_retriever.ingest.parsers.PARSE_CACHE_DIR", tmp_path / "parse_cache")
    monkeypatch.setattr("adaptive_retriever.ingest.parsers.CANONICAL_DIR", tmp_path / "canonical")
    monkeypatch.setattr("adaptive_retriever.llm.get_llm_client", lambda cfg=None: StubLLMClient())

    # 4. Patch VectorStore to use in-memory qdrant
    with patch("adaptive_retriever.vectorstore.QdrantClient") as mock_qdrant:
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance

        stats = run_ingestion_pipeline(manifest_path=manifest_p)

        assert stats["ingested_documents"] == 3
        assert stats["total_chunks"] > 0
        assert stats["total_representations"] > 0
        assert report_p.exists()
