"""Check class confidence distribution across all 421 corpus documents in profiled reality."""

import json
from collections import defaultdict
from pathlib import Path
from adaptive_retriever.ingest.profiler import profile

MANIFEST_PATH = Path("data/corpus/MANIFEST.jsonl")
OUT_PATH = Path("data/confidence_summary.txt")


def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    class_confidences = defaultdict(list)
    all_confs = []

    for rec in records:
        p = Path(rec["doc_path"])
        if not p.exists():
            continue
        fp = profile(p)
        conf = fp.class_confidence
        cls_val = fp.doc_class.value
        class_confidences[cls_val].append(conf)
        all_confs.append(conf)

    lines = []
    lines.append(f"Total profiled: {len(all_confs)}")
    lines.append(f"Overall Confidence Range: [{min(all_confs):.2f}, {max(all_confs):.2f}] | Mean: {sum(all_confs)/len(all_confs):.4f}\n")

    lines.append(f"{'DocClass':<20}{'Count':>10}{'Min Conf':>12}{'Mean Conf':>12}{'Max Conf':>12}")
    lines.append("-" * 66)
    for cls_val, confs in class_confidences.items():
        c_min = min(confs)
        c_mean = sum(confs) / float(len(confs))
        c_max = max(confs)
        lines.append(f"{cls_val:<20}{len(confs):>10}{c_min:>12.2f}{c_mean:>12.4f}{c_max:>12.2f}")
    lines.append("-" * 66)

    output_text = "\n".join(lines)
    OUT_PATH.write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()


