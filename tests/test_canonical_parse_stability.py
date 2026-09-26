"""Guards spec §B3: Canonical parse hash stability test.
Fails loudly if any canonical parse is altered or regenerated.
"""

import hashlib
import json
from pathlib import Path
import pytest

CANONICAL_DIR = Path(__file__).resolve().parent.parent / "data" / "canonical"
HASHES_PATH = Path(__file__).resolve().parent.parent / "data" / "canonical_hashes.json"


MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus" / "MANIFEST.jsonl"


@pytest.mark.skipif(not HASHES_PATH.exists() or not CANONICAL_DIR.exists(), reason="Corpus data not present")
def test_canonical_parse_stability():
    assert HASHES_PATH.exists(), f"Canonical hashes manifest missing at {HASHES_PATH}"

    with open(HASHES_PATH, "r", encoding="utf-8") as f:
        hashes = json.load(f)

    if "doc_hashes" in hashes:
        hashes = hashes["doc_hashes"]

    assert len(hashes) > 0, "Canonical hashes manifest is empty!"

    violations = []
    for doc_id, meta in hashes.items():
        if not isinstance(meta, dict):
            continue
        txt_file = CANONICAL_DIR / f"{doc_id}.txt"
        if not txt_file.exists():
            violations.append(f"Missing canonical parse file: {txt_file}")
            continue

        current_text = txt_file.read_text(encoding="utf-8")
        current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()

        if current_hash != meta["text_hash"]:
            violations.append(
                f"Canonical hash mismatch for doc_id {doc_id}!\n"
                f"Expected hash: {meta['text_hash']}\n"
                f"Current hash:  {current_hash}"
            )

    assert not violations, "Canonical parse stability violation:\n" + "\n".join(violations)


@pytest.mark.skipif(not MANIFEST_PATH.exists() or not CANONICAL_DIR.exists(), reason="Corpus data not present")
def test_no_orphaned_canonical_parses():
    assert MANIFEST_PATH.exists(), f"Manifest file missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    valid_doc_ids = set()
    for rec in records:
        doc_path = Path(rec["doc_path"])
        if doc_path.exists():
            d_id = hashlib.sha256(doc_path.read_bytes()).hexdigest()[:16]
            valid_doc_ids.add(d_id)
        elif "doc_id" in rec:
            valid_doc_ids.add(rec["doc_id"])

    canonical_files = list(CANONICAL_DIR.glob("*.txt"))
    orphans = [f for f in canonical_files if f.stem not in valid_doc_ids]

    assert not orphans, f"Found {len(orphans)} orphaned canonical parse files not in manifest: {[f.name for f in orphans[:5]]}"
    assert len(canonical_files) == len(valid_doc_ids), (
        f"Mismatch between canonical parse file count ({len(canonical_files)}) "
        f"and valid manifest document ID count ({len(valid_doc_ids)})"
    )
