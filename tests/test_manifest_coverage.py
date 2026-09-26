"""Coverage test for data/corpus/MANIFEST.jsonl (Addendum A1, E, F)."""

import json
from collections import Counter
from pathlib import Path
import pytest

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus" / "MANIFEST.jsonl"


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Corpus data not present")
def test_manifest_coverage():
    assert MANIFEST_PATH.exists(), f"MANIFEST.jsonl does not exist at {MANIFEST_PATH}"

    records = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) >= 400, f"Expected at least 400 documents in manifest (spec §8.1), found {len(records)}"

    classes = [r.get("expected_doc_class") for r in records]
    class_counts = Counter(classes)

    print(f"Manifest document count: {len(records)}")
    print(f"DocClass distribution: {dict(class_counts)}")

    required_classes = ["digital_prose", "table_heavy", "scanned", "html", "code", "transcript"]
    for doc_class in required_classes:
        assert class_counts[doc_class] >= 40, (
            f"DocClass '{doc_class}' has only {class_counts[doc_class]} documents "
            f"(minimum 40 required per spec §8.1)."
        )

