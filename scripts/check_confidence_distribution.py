"""Fast parallel script to check class confidence distribution and low-confidence logging."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from adaptive_retriever.ingest.profiler import profile, LOW_CONFIDENCE_LOG

MANIFEST_PATH = Path("data/corpus/MANIFEST.jsonl")


def profile_doc(rec):
    p = Path(rec["doc_path"])
    if not p.exists():
        return None
    fp = profile(p)
    return fp.class_confidence, fp.doc_class.value


def main():
    if not MANIFEST_PATH.exists():
        print("Manifest missing.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    confidences = []
    low_conf_count = 0

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(profile_doc, rec) for rec in records]
        for f in as_completed(futures):
            res = f.result()
            if res:
                conf, cls_val = res
                confidences.append(conf)
                if conf < 0.4:
                    low_conf_count += 1

    print(f"Total documents profiled: {len(confidences)}")
    print(f"Min Confidence:     {min(confidences):.2f}")
    print(f"Max Confidence:     {max(confidences):.2f}")
    print(f"Average Confidence: {sum(confidences)/len(confidences):.2f}")
    print(f"Low Confidence (<0.4) Count: {low_conf_count}")
    print(f"Low confidence log file exists: {LOW_CONFIDENCE_LOG.exists()}")
    if LOW_CONFIDENCE_LOG.exists():
        log_lines = LOW_CONFIDENCE_LOG.read_text(encoding="utf-8").splitlines()
        print(f"Low confidence log line count: {len(log_lines)}")


if __name__ == "__main__":
    main()
