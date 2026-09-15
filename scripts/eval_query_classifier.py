"""Task A.5: Query Classifier Evaluation.

Evaluates profile_query over all 169 non-rejected items in data/eval/eval_set.jsonl.
Computes accuracy, 6x6 confusion matrix, per-class precision/recall/support,
class_confidence distributions for correct vs incorrect predictions,
top 5 common confusions with example queries, and compound vs multihop cross-confusion count.
Saves to reports/classifier_eval.json and prints matrix to stdout.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from adaptive_retriever.models import EvalItem, QueryClass
from adaptive_retriever.query.profiler import profile_query


def main():
    eval_path = Path("data/eval/eval_set.jsonl")
    records = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid_items = [EvalItem(**r) for r in records if r.get("provenance") != "synthetic_rejected"]

    classes = [qc.value for qc in QueryClass]
    conf_matrix = {pred: {exp: 0 for exp in classes} for pred in classes}

    correct_confidences = []
    incorrect_confidences = []

    per_item_results = []
    confusions_list = []

    for item in valid_items:
        fp = profile_query(item.query)
        pred_cls = fp.query_class.value if hasattr(fp.query_class, "value") else str(fp.query_class)
        exp_cls = item.query_class_expected.value if hasattr(item.query_class_expected, "value") else str(item.query_class_expected)
        conf = float(fp.class_confidence)

        conf_matrix[pred_cls][exp_cls] += 1

        is_correct = (pred_cls == exp_cls)
        if is_correct:
            correct_confidences.append(conf)
        else:
            incorrect_confidences.append(conf)
            confusions_list.append({
                "item_id": item.item_id,
                "query": item.query,
                "predicted": pred_cls,
                "expected": exp_cls,
                "confidence": conf,
            })

        per_item_results.append({
            "item_id": item.item_id,
            "query": item.query,
            "predicted": pred_cls,
            "expected": exp_cls,
            "confidence": conf,
            "correct": is_correct,
        })

    # Overall accuracy
    correct_total = len(correct_confidences)
    total_eval = len(valid_items)
    accuracy = float(correct_total / total_eval) if total_eval > 0 else 0.0

    # Per-class Precision, Recall, Support
    per_class_metrics = {}
    for c in classes:
        tp = conf_matrix[c][c]
        fp_count = sum(conf_matrix[c][exp] for exp in classes if exp != c)
        fn_count = sum(conf_matrix[pred][c] for pred in classes if pred != c)
        support = sum(conf_matrix[pred][c] for pred in classes)

        precision = float(tp / (tp + fp_count)) if (tp + fp_count) > 0 else 0.0
        recall = float(tp / (tp + fn_count)) if (tp + fn_count) > 0 else 0.0

        per_class_metrics[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "support": support,
            "tp": tp,
            "fp": fp_count,
            "fn": fn_count,
        }

    # Confidence distribution
    def get_stats(vals):
        if not vals:
            return {"mean": 0.0, "std": 0.0, "p50": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(vals, dtype=float)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    correct_stats = get_stats(correct_confidences)
    incorrect_stats = get_stats(incorrect_confidences)

    # Common confusions breakdown
    confusion_pair_counts: Dict[str, List[dict]] = {}
    for conf in confusions_list:
        pair_key = f"{conf['predicted']} (pred) vs {conf['expected']} (exp)"
        if pair_key not in confusion_pair_counts:
            confusion_pair_counts[pair_key] = []
        confusion_pair_counts[pair_key].append(conf)

    sorted_confusions = sorted(confusion_pair_counts.items(), key=lambda x: len(x[1]), reverse=True)
    top_5_confusions = []
    for pair_key, items_list in sorted_confusions[:5]:
        top_5_confusions.append({
            "pair": pair_key,
            "count": len(items_list),
            "example_query": items_list[0]["query"],
            "example_item_id": items_list[0]["item_id"],
        })

    # Specific flag: compound vs multihop
    exp_compound_pred_multihop = sum(1 for conf in confusions_list if conf["expected"] == "compound" and conf["predicted"] == "multihop")
    exp_multihop_pred_compound = sum(1 for conf in confusions_list if conf["expected"] == "multihop" and conf["predicted"] == "compound")

    report_data = {
        "meta": {
            "total_items_evaluated": total_eval,
            "overall_accuracy": round(accuracy, 4),
            "correct_count": correct_total,
            "incorrect_count": len(incorrect_confidences),
        },
        "confusion_matrix_predicted_x_expected": conf_matrix,
        "per_class_metrics": per_class_metrics,
        "confidence_distribution": {
            "correct": correct_stats,
            "incorrect": incorrect_stats,
            "separates_correct_vs_incorrect": bool(abs(correct_stats["mean"] - incorrect_stats["mean"]) > 0.05),
        },
        "top_5_common_confusions": top_5_confusions,
        "comparison_cross_confusion": {
            "expected_compound_predicted_multihop": exp_compound_pred_multihop,
            "expected_multihop_predicted_compound": exp_multihop_pred_compound,
        },
    }

    # Save to reports/classifier_eval.json
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "classifier_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Saved classifier evaluation report to {out_file}\n")

    # Print 6x6 Confusion Matrix to stdout
    print("=" * 115)
    print(" QUERY CLASSIFIER EVALUATION: 6x6 CONFUSION MATRIX (Predicted [rows] x Expected [cols])")
    print(f" (Total Evaluated: {total_eval} | Accuracy: {accuracy*100:.2f}%)")
    print("=" * 115)
    pred_exp_title = "Predicted \\ Expected"
    header_str = f"{pred_exp_title:<22} | " + " | ".join(f"{c[:12]:<12}" for c in classes)
    print(header_str)
    print("-" * 115)

    for pred in classes:
        row_str = f"{pred:<22} | " + " | ".join(f"{conf_matrix[pred][exp]:<12}" for exp in classes)
        print(row_str)

    print("=" * 115 + "\n")
    print("PER-CLASS METRICS:")
    print(f"{'Class':<18} | {'Precision':<10} | {'Recall':<10} | {'Support':<10}")
    print("-" * 55)
    for c in classes:
        m = per_class_metrics[c]
        print(f"{c:<18} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['support']:<10}")

    print("\n" + "=" * 55)
    print(f"CONFIDENCE DISTRIBUTION:")
    print(f"Correct predictions   : Mean = {correct_stats['mean']:.4f}, Std = {correct_stats['std']:.4f}, p50 = {correct_stats['p50']:.4f}")
    print(f"Incorrect predictions : Mean = {incorrect_stats['mean']:.4f}, Std = {incorrect_stats['std']:.4f}, p50 = {incorrect_stats['p50']:.4f}")
    print(f"Separates Correct/Incorrect: {report_data['confidence_distribution']['separates_correct_vs_incorrect']}")
    print("=" * 55 + "\n")

    print("COMPARISON QUERY CROSS-CONFUSION:")
    print(f"  Expected Compound -> Predicted Multihop: {exp_compound_pred_multihop}")
    print(f"  Expected Multihop -> Predicted Compound: {exp_multihop_pred_compound}")


if __name__ == "__main__":
    main()
