#!/usr/bin/env python3
"""Report Verification Script (Task R-02a).

Verifies 7 data integrity checks on report JSON files.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

class VerificationError(ValueError):
    """Raised when a report fails verification."""
    pass

def parse_ci_str(ci_str: str) -> Tuple[float, float]:
    """Parse '[low, high]' string into float tuple."""
    if "[" in ci_str and "]" in ci_str:
        inside = ci_str.split("[")[1].split("]")[0]
        parts = [p.strip().lstrip("+") for p in inside.split(",")]
        return float(parts[0]), float(parts[1])
    raise ValueError(f"Invalid CI string: {ci_str}")

def check_range(data: Dict[str, Any]) -> List[str]:
    """Check 1: RANGE - metric values & CI bounds within [0, 1]."""
    warnings = []
    
    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                if isinstance(v, (int, float)):
                    k_lower = k.lower()
                    if any(m in k_lower for m in ["recall", "ndcg", "mrr", "rate", "pct"]):
                        val = float(v)
                        if "pct" in k_lower and val > 1.0 and val <= 100.0:
                            val = val / 100.0
                        min_b = -1.0 if "diff" in k_lower else 0.0
                        if val < min_b - 1e-6 or val > 1.0 + 1e-6:
                            raise VerificationError(f"RANGE violation at '{new_path}': value {v} outside [{min_b}, 1]")
                elif isinstance(v, str) and "[" in v and "]" in v:
                    k_lower = k.lower()
                    if any(m in k_lower for m in ["arm", "diff", "ci", "metric", "recall", "ndcg", "mrr"]):
                        try:
                            low, high = parse_ci_str(v)
                            min_b = -1.0 if "diff" in k_lower else 0.0
                            if low < min_b - 1e-6 or low > 1.0 + 1e-6 or high < min_b - 1e-6 or high > 1.0 + 1e-6:
                                raise VerificationError(f"RANGE violation at '{new_path}': CI [{low}, {high}] outside [{min_b}, 1]")
                        except VerificationError:
                            raise
                        except (ValueError, IndexError):
                            pass
                walk(v, new_path)
        elif isinstance(obj, list):
            for i, elem in enumerate(obj):
                walk(elem, f"{path}[{i}]")

    walk(data)
    return warnings

def check_arithmetic(data: Dict[str, Any]) -> List[str]:
    """Check 2: ARITHMETIC - reported diff == arm_b - arm_a within 1e-6."""
    warnings = []
    
    def check_table(rows):
        for row in rows:
            if not isinstance(row, dict):
                continue
            mean_a = row.get("mean_A", row.get("mean_a", None))
            mean_b = row.get("mean_B", row.get("mean_b", None))
            mean_c = row.get("mean_C", row.get("mean_c", None))
            diff_cb = row.get("diff_CB", row.get("diff_cb", row.get("diff", None)))
            
            if mean_b is not None and mean_c is not None and diff_cb is not None:
                expected_diff = float(mean_c) - float(mean_b)
                if abs(float(diff_cb) - expected_diff) >= 1e-6:
                    raise VerificationError(
                        f"ARITHMETIC violation for metric '{row.get('metric')}': "
                        f"reported diff {diff_cb} != {mean_c} - {mean_b} ({expected_diff:.6f})"
                    )
            elif mean_a is not None and mean_b is not None and diff_cb is not None:
                expected_diff = float(mean_b) - float(mean_a)
                if abs(float(diff_cb) - expected_diff) >= 1e-6:
                    raise VerificationError(
                        f"ARITHMETIC violation for metric '{row.get('metric')}': "
                        f"reported diff {diff_cb} != {mean_b} - {mean_a} ({expected_diff:.6f})"
                    )
                    
    for k in ["table_full", "table_powered", "table", "metrics"]:
        if k in data and isinstance(data[k], list):
            check_table(data[k])
            
    return warnings

def check_subgroup_sums(data: Dict[str, Any]) -> List[str]:
    """Check 3: SUBGROUP SUMS - weighted subgroup mean equals reported total within 1e-6."""
    warnings = []
    subgroups = data.get("subgroups", data.get("subgroup_breakdown", None))
    if isinstance(subgroups, dict):
        total_n = subgroups.get("total_n", subgroups.get("n_total", None))
        total_mean = subgroups.get("total_mean", None)
        parts = subgroups.get("groups", subgroups.get("parts", []))
        
        if total_n and total_mean is not None and parts:
            calc_sum = sum(p["n"] * p["mean"] for p in parts)
            calc_mean = calc_sum / float(total_n)
            if abs(calc_mean - float(total_mean)) >= 1e-6:
                raise VerificationError(
                    f"SUBGROUP SUMS violation: weighted subgroup mean {calc_mean:.6f} "
                    f"!= reported total mean {total_mean:.6f}"
                )
    return warnings

def check_quantisation(data: Dict[str, Any]) -> List[str]:
    """Check 4: QUANTISATION - item count diff * N and flag if 0 or 1."""
    notes = []
    
    def inspect_diffs(rows, N):
        for row in rows:
            if not isinstance(row, dict):
                continue
            diff = row.get("diff_CB", row.get("diff_cb", row.get("diff", None)))
            metric = row.get("metric", "metric")
            if diff is not None and N > 0:
                implied_items = round(abs(float(diff)) * N)
                exact_val = abs(float(diff)) * N
                notes.append(f"QUANTISATION report for '{metric}': diff={diff:+.6f} * N={N} -> implied item count = {exact_val:.4f} (~{implied_items} items)")
                if implied_items in (0, 1):
                    notes.append(f"QUANTISATION FLAG: '{metric}' diff implies only {implied_items} query item delta (N={N})!")
                    
    N = data.get("full_queries", data.get("total_queries", data.get("N", 169)))
    for k in ["table_full", "table"]:
        if k in data and isinstance(data[k], list):
            inspect_diffs(data[k], N)
            
    return notes

def check_count_deltas(data: Dict[str, Any]) -> List[str]:
    """Check 5: COUNT DELTAS - after_count - before_count == stated_added."""
    warnings = []
    deltas = data.get("count_deltas", data.get("counts", None))
    if isinstance(deltas, dict):
        before = deltas.get("before", deltas.get("before_count", None))
        after = deltas.get("after", deltas.get("after_count", None))
        added = deltas.get("added", deltas.get("stated_added", None))
        if before is not None and after is not None and added is not None:
            if (after - before) != added:
                raise VerificationError(
                    f"COUNT DELTAS violation: after ({after}) - before ({before}) = {after - before} != stated added ({added})"
                )
    return warnings

def check_rate_plausibility(data: Dict[str, Any]) -> List[str]:
    """Check 6: RATE PLAUSIBILITY - implied per-item time within 10x of benchmark."""
    warnings = []
    bench = data.get("benchmark", data.get("timing_benchmark", None))
    if isinstance(bench, dict):
        count = bench.get("count", None)
        wall_clock = bench.get("wall_clock_sec", None)
        known_rate = bench.get("known_sec_per_item", None)
        if count and wall_clock and known_rate:
            implied_rate = wall_clock / float(count)
            ratio = implied_rate / float(known_rate) if known_rate > 0 else 1.0
            if ratio < 0.1 or ratio > 10.0:
                raise VerificationError(
                    f"RATE PLAUSIBILITY violation: implied rate {implied_rate:.4f} s/item "
                    f"is >10x different from benchmark {known_rate:.4f} s/item (ratio {ratio:.2f})"
                )
    return warnings

def check_cross_report_drift(data: Dict[str, Any], prev_data: Dict[str, Any]) -> List[str]:
    """Check 7: CROSS-REPORT DRIFT - print list of changed metric values."""
    drift_lines = ["--- DRIFT SECTION ---"]
    
    def extract_metrics(d):
        metrics = {}
        for k in ["table_full", "table", "table_powered"]:
            if k in d and isinstance(d[k], list):
                for row in d[k]:
                    if isinstance(row, dict) and "metric" in row:
                        m_name = f"{k}.{row['metric']}"
                        for val_k in ["Arm_A", "Arm_B", "Arm_C", "mean_C", "mean_B"]:
                            if val_k in row:
                                metrics[f"{m_name}.{val_k}"] = row[val_k]
        return metrics

    curr_m = extract_metrics(data)
    prev_m = extract_metrics(prev_data)
    
    drift_found = False
    for k, val in curr_m.items():
        if k in prev_m and prev_m[k] != val:
            drift_found = True
            drift_lines.append(f"  {k}: prev = {prev_m[k]}  -->  curr = {val}")
            
    if not drift_found:
        drift_lines.append("  No metric drift detected between reports.")
        
    return drift_lines

def verify_report(json_path: Path, prev_json_path: Path = None) -> bool:
    """Run all verification checks on a report JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"\n==================================================")
    print(f"VERIFYING REPORT: {json_path}")
    print(f"==================================================")
    
    # Run 6 failure-gated checks
    check_range(data)
    check_arithmetic(data)
    check_subgroup_sums(data)
    quant_notes = check_quantisation(data)
    check_count_deltas(data)
    check_rate_plausibility(data)
    
    for note in quant_notes:
        print(f"[QUANTISATION] {note}")
        
    print("ALL 6 HARD CHECKS PASSED SUCCESSFULLY.")
    
    if prev_json_path and prev_json_path.exists():
        with open(prev_json_path, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        drift_lines = check_cross_report_drift(data, prev_data)
        for line in drift_lines:
            print(line)
            
    print("==================================================\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify report JSON integrity.")
    parser.add_argument("report_json", type=Path, help="Path to report JSON")
    parser.add_argument("--prev", type=Path, default=None, help="Previous report JSON for drift check")
    args = parser.parse_args()

    if not args.report_json.exists():
        print(f"Error: file {args.report_json} does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        verify_report(args.report_json, args.prev)
        sys.exit(0)
    except VerificationError as err:
        print(f"\nVERIFICATION FAILED LOUDLIES: {err}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\nUNEXPECTED ERROR: {err}\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
