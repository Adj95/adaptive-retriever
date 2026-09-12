"""Comprehensive profiler & eval suite verification script."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from adaptive_retriever.ingest.profiler import profile

sys.stdout.reconfigure(encoding="utf-8")
MANIFEST_PATH = Path("data/corpus/MANIFEST.jsonl")
EVAL_PATH = Path("data/eval/eval_set.jsonl")


def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f if line.strip()]

    matrix = defaultdict(Counter)
    class_confs = defaultdict(list)
    languages = Counter()
    all_confs = []

    for rec in manifest:
        expected = rec["expected_doc_class"]
        p = Path(rec["doc_path"])
        if not p.exists():
            continue
        fp = profile(p)
        actual = fp.doc_class.value
        matrix[expected][actual] += 1
        class_confs[actual].append(fp.class_confidence)
        all_confs.append(fp.class_confidence)
        languages[fp.language] += 1

    doc_classes = ["digital_prose", "table_heavy", "scanned", "html", "code", "transcript"]
    header = f"{'Expected':<20}" + "".join(f"{act:>13}" for act in doc_classes)

    eval_items = []
    if EVAL_PATH.exists():
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            eval_items = [json.loads(line) for line in f if line.strip()]

    agg_items = [item for item in eval_items if item.get("query_class") == "aggregation"]
    filt_items = [item for item in eval_items if item.get("query_class") == "filterable"]

    agg_lines = []
    for idx, item in enumerate(agg_items[:5], 1):
        snip = item['relevant_passages'][0]['passage_text'][:120].replace('\n', ' ')
        agg_lines.append(f"\n[Agg #{idx}] ID: {item['item_id']} | Passages: {len(item['relevant_passages'])}\nQuery:   {item['query']}\nPassage: {snip}...")

    filt_lines = []
    for idx, item in enumerate(filt_items[:5], 1):
        snip = item['relevant_passages'][0]['passage_text'][:120].replace('\n', ' ')
        filt_lines.append(f"\n[Filt #{idx}] ID: {item['item_id']} | Passages: {len(item['relevant_passages'])}\nQuery:   {item['query']}\nPassage: {snip}...")

    output_text = "\n".join([
        "==================================================================",
        " 1. CONFUSION MATRIX (Expected vs Profiled Actual)",
        "==================================================================",
        header,
        "-" * len(header),
    ] + [
        f"{exp:<20}" + "".join(f"{matrix[exp][act]:>13}" for act in doc_classes) for exp in doc_classes
    ] + [
        "-" * len(header),
        "\n==================================================================",
        " 2. CONTINUOUS CLASS CONFIDENCE DISTRIBUTION",
        "==================================================================",
        f"Overall Confidence Range: [{min(all_confs):.4f}, {max(all_confs):.4f}] | Mean: {sum(all_confs)/len(all_confs):.4f}",
        f"{'DocClass':<20}{'Count':>10}{'Min Conf':>12}{'Mean Conf':>12}{'Max Conf':>12}",
        "-" * 66,
    ] + [
        f"{cls:<20}{len(class_confs[cls]):>10}{min(class_confs[cls]):>12.4f}{sum(class_confs[cls])/len(class_confs[cls]):>12.4f}{max(class_confs[cls]):>12.4f}"
        for cls in doc_classes if class_confs[cls]
    ] + [
        "-" * 66,
        "\n==================================================================",
        " 3. CORPUS LANGUAGE DISTRIBUTION",
        "==================================================================",
    ] + [
        f"Language '{lang}': {count} documents ({count/float(len(manifest))*100:.1f}%)"
        for lang, count in languages.most_common()
    ] + [
        "\n==================================================================",
        " 4. EVAL SAMPLE: AGGREGATION QUERIES (5 items)",
        "==================================================================",
    ] + agg_lines + [
        "\n==================================================================",
        " 5. EVAL SAMPLE: FILTERABLE QUERIES (5 items)",
        "==================================================================",
    ] + filt_lines)

    Path("data/verification_output.txt").write_text(output_text, encoding="utf-8")
    print(output_text, flush=True)


if __name__ == "__main__":
    main()

