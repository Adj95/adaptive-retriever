"""Instant script to compute and display the confusion matrix of expected vs profiled DocClass."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_retriever.ingest.profiler import profile

MANIFEST_PATH = Path("data/corpus/MANIFEST.jsonl")
HASHES_PATH = Path("data/canonical_hashes.json")


def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest file {MANIFEST_PATH} missing.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    hashes = {}
    if HASHES_PATH.exists():
        with open(HASHES_PATH, "r", encoding="utf-8") as f:
            hashes = json.load(f)

    # Build path to hash mapping
    path_to_profile = {}
    for doc_id, meta in hashes.items():
        path_to_profile[meta["source_path"]] = meta["doc_class"]

    doc_classes = ["digital_prose", "table_heavy", "scanned", "html", "code", "transcript", "unknown"]
    confusion_matrix = defaultdict(Counter)
    profiled_counts = Counter()

    agreement_matches = 0
    total = len(records)

    for rec in records:
        doc_path = rec["doc_path"]
        expected = rec.get("expected_doc_class", "unknown")

        actual = path_to_profile.get(doc_path)
        if not actual:
            p = Path(doc_path)
            if p.exists():
                fp = profile(p)
                actual = fp.doc_class.value
            else:
                actual = "unknown"

        confusion_matrix[expected][actual] += 1
        profiled_counts[actual] += 1
        if expected == actual:
            agreement_matches += 1

    print("\n=================== DOC_CLASS CONFUSION MATRIX ===================")
    header_title = "Expected \\ Profiled"
    print(f"{header_title:<20}" + "".join(f"{c:>15}" for c in doc_classes))
    print("-" * 125)

    for exp_cls in ["digital_prose", "table_heavy", "scanned", "html", "code", "transcript"]:
        row_str = f"{exp_cls:<20}"
        for act_cls in doc_classes:
            cnt = confusion_matrix[exp_cls][act_cls]
            row_str += f"{cnt:>15}"
        print(row_str)

    print("-" * 125)
    print("\n=================== PROFILED REALITY SUMMARY ===================")
    print(f"Overall Agreement Rate: {agreement_matches / float(max(1, total)) * 100.0:.1f}% ({agreement_matches}/{total})")
    print(f"Profiled Class Distribution: {dict(profiled_counts)}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
