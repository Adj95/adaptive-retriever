"""Post-hoc aggregation for Experiment 006.

Reads reports/experiment_006_raw.jsonl and computes aggregate metrics,
bootstrapped 95% CIs, paired difference CIs against Arm 1, and multiplicity metadata.
Outputs to reports/experiment_006.json.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List

from adaptive_retriever.eval.metrics import bootstrap_ci, bootstrap_difference_ci


def aggregate_experiment_006():
    raw_jsonl_path = Path("reports/experiment_006_raw.jsonl")
    if not raw_jsonl_path.exists():
        raise FileNotFoundError(f"Missing {raw_jsonl_path}")

    records = []
    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # De-duplicate by item_idx keeping latest
    records_by_idx = {}
    for r in records:
        records_by_idx[r["item_idx"]] = r

    unique_records = [records_by_idx[idx] for idx in sorted(records_by_idx.keys())]
    print(f"Loaded {len(unique_records)} unique item records from {raw_jsonl_path}")

    arms_def = [
        "Arm 1: generation_alone",
        "Arm 2: +reorder",
        "Arm 3: +compression",
        "Arm 4: +both",
    ]

    arm_results: Dict[str, Dict[str, List[float]]] = {
        arm_name: {
            "rouge_1": [],
            "rouge_2": [],
            "rouge_l": [],
            "bleurt_20": [],
            "has_citation": [],
            "unsupported_pct": [],
            "tot_ms": [],
        }
        for arm_name in arms_def
    }

    for rec in unique_records:
        for arm_name in arms_def:
            arm_data = rec["arms"][arm_name]
            for key in arm_results[arm_name]:
                arm_results[arm_name][key].append(float(arm_data[key]))

    base_arm_name = "Arm 1: generation_alone"
    base_metrics = arm_results[base_arm_name]
    metric_keys = ["rouge_1", "rouge_2", "rouge_l", "bleurt_20", "has_citation", "unsupported_pct", "tot_ms"]

    summary = {}
    for arm_name in arms_def:
        arm_summary = {}
        for key in metric_keys:
            vals = arm_results[arm_name][key]
            mean_val, ci_lo, ci_hi = bootstrap_ci(vals, n=1000, seed=42)

            metric_entry = {
                "mean": round(float(mean_val), 4),
                "ci_95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
            }

            if arm_name != base_arm_name:
                base_vals = base_metrics[key]
                diff_mean, d_lo, d_hi = bootstrap_difference_ci(vals, base_vals, n=1000, seed=42)
                metric_entry["diff_vs_baseline"] = {
                    "mean_diff": round(float(diff_mean), 4),
                    "ci_95": [round(float(d_lo), 4), round(float(d_hi), 4)],
                }

            arm_summary[key] = metric_entry

        summary[arm_name] = arm_summary

    multiplicity_count = (len(arms_def) - 1) * len(metric_keys)  # 3 comparative arms * 7 metrics = 21 hypothesis tests

    output_report = {
        "meta": {
            "experiment_id": "experiment_006",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_items": len(unique_records),
            "multiplicity_count": multiplicity_count,
            "multiplicity_explanation": f"{len(arms_def) - 1} comparative arms evaluated against Baseline ({base_arm_name}) across {len(metric_keys)} metrics = {multiplicity_count} hypothesis tests.",
        },
        "arms": summary,
    }

    out_path = Path("reports/experiment_006.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)

    print(f"Successfully generated {out_path}!")
    print(json.dumps(output_report, indent=2))


if __name__ == "__main__":
    aggregate_experiment_006()
