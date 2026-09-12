"""Determinism test for Document Profiler (spec §9.1)."""

import pytest
from pathlib import Path
from adaptive_retriever.ingest.profiler import profile


def test_profiler_determinism(tmp_path):
    sample_file = tmp_path / "sample_doc.html"
    sample_file.write_text("<html><body><h1>Title</h1><table><tr><td>Data</td></tr></table></body></html>", encoding="utf-8")

    fp1 = profile(sample_file)
    fp2 = profile(sample_file)

    assert fp1.model_dump_json() == fp2.model_dump_json()
    assert fp1.doc_class == fp2.doc_class
    assert fp1.class_confidence == fp2.class_confidence


def test_profiler_low_confidence_logging(tmp_path):
    from adaptive_retriever.ingest.profiler import LOW_CONFIDENCE_LOG, DocumentFingerprint, log_low_confidence_fingerprint
    fp = DocumentFingerprint(
        source_path=str(tmp_path / "low_conf.txt"),
        mime_type="text/plain",
        char_count=100,
        has_extractable_text=True,
        doc_class="unknown",
        class_confidence=0.20,
    )
    log_low_confidence_fingerprint(fp)
    assert LOW_CONFIDENCE_LOG.exists()
    lines = LOW_CONFIDENCE_LOG.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0


