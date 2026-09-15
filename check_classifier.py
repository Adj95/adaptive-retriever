"""Measure the query classifier against the evaluation set's ground truth.

Six query classes decide the fusion weights, the transformation strategy, and
later the escalation path. Nothing has ever measured how often the classifier
is right. Every eval item carries `query_class_expected`, so this is free.

Reads only. Writes one report file. Does NOT tune anything - tuning the
classifier on the eval set would contaminate every later experiment.

    python check_classifier.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_retriever.models import QueryClass
from adaptive_retriever.query.profiler import profile_query

EVAL = Path("data/eval/eval_set.jsonl")
OUT = Path("reports/classifier_eval.json")
CLASSES = [qc.value for qc in QueryClass]

if not EVAL.exists():
    sys.exit(f"Eval set not found at {EVAL}. Run this from the repo root.")

rows = [json.loads(l) for l in EVAL.open(encoding="utf-8") if l.strip()]

pairs = []          # (expected, predicted, confidence, query, item_id)
skipped = Counter()
for r in rows:
    if r.get("provenance") == "synthetic_rejected":
        skipped["synthetic_rejected"] += 1
        continue
    expected = r.get("query_class_expected")
    if not expected:
        skipped["no_expected_label"] += 1
        continue
    q = r.get("query") or ""
    if not q.strip():
        skipped["empty_query"] += 1
        continue
    fp = profile_query(q)
    predicted = fp.query_class.value if hasattr(fp.query_class, "value") else str(fp.query_class)
    pairs.append((str(expected), predicted, float(fp.class_confidence), q, r.get("item_id", "?")))

if not pairs:
    sys.exit("No labelled items to score.")

n = len(pairs)
correct = sum(1 for e, p, _, _, _ in pairs if e == p)
accuracy = correct / n

print("=" * 78)
print("QUERY CLASSIFIER EVALUATION")
print("=" * 78)
print(f"  items scored : {n}")
print(f"  skipped      : {dict(skipped)}")
print(f"  accuracy     : {accuracy:.4f}  ({correct}/{n})")

# a useless-but-honest reference point: always predict the most common class
majority = Counter(e for e, _, _, _, _ in pairs).most_common(1)[0]
print(f"  majority-class baseline: {majority[1] / n:.4f}  (always predict '{majority[0]}')")
if accuracy <= majority[1] / n:
    print("  WARNING: the classifier is no better than always guessing one class.")

# ------------------------------------------------------- confusion matrix ---
matrix = {e: Counter() for e in CLASSES}
for e, p, _, _, _ in pairs:
    matrix.setdefault(e, Counter())[p] += 1

print()
print("=" * 78)
print("CONFUSION MATRIX   rows = expected, columns = predicted")
print("=" * 78)
w = max(len(c) for c in CLASSES) + 1
print(" " * w + "".join(f"{c[:8]:>10}" for c in CLASSES) + f"{'total':>10}")
for e in CLASSES:
    row = matrix.get(e, Counter())
    total = sum(row.values())
    cells = "".join(f"{row.get(p, 0):>10}" for p in CLASSES)
    print(f"{e:<{w}}{cells}{total:>10}")

# ---------------------------------------------------------- per-class P/R ---
print()
print("=" * 78)
print("PER-CLASS PRECISION / RECALL / SUPPORT")
print("=" * 78)
print(f"{'class':<16}{'precision':>11}{'recall':>9}{'F1':>8}{'support':>9}{'predicted':>11}")
per_class = {}
for c in CLASSES:
    tp = matrix.get(c, Counter()).get(c, 0)
    support = sum(matrix.get(c, Counter()).values())
    predicted_n = sum(matrix.get(e, Counter()).get(c, 0) for e in CLASSES)
    prec = tp / predicted_n if predicted_n else 0.0
    rec = tp / support if support else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": support, "predicted": predicted_n}
    print(f"{c:<16}{prec:>11.4f}{rec:>9.4f}{f1:>8.4f}{support:>9}{predicted_n:>11}")

never = [c for c in CLASSES if per_class[c]["predicted"] == 0]
absent = [c for c in CLASSES if per_class[c]["support"] == 0]
if never:
    print(f"\n  NOTE: never predicted by the classifier: {never}")
if absent:
    print(f"  NOTE: no eval items expect these classes: {absent}")

# -------------------------------------------------- is confidence useful? ---
conf_ok = [c for e, p, c, _, _ in pairs if e == p]
conf_bad = [c for e, p, c, _, _ in pairs if e != p]


def stats(v):
    if not v:
        return None
    s = sorted(v)
    return {"n": len(v), "mean": round(sum(s) / len(s), 4),
            "p50": round(s[len(s) // 2], 4), "min": round(s[0], 4), "max": round(s[-1], 4)}


print()
print("=" * 78)
print("IS class_confidence A USEFUL SIGNAL?")
print("=" * 78)
sc, sw = stats(conf_ok), stats(conf_bad)
print(f"  when CORRECT   : {sc}")
print(f"  when WRONG     : {sw}")
if sc and sw:
    gap = sc["mean"] - sw["mean"]
    print(f"  mean gap       : {gap:+.4f}")
    if abs(gap) < 0.03:
        print("  VERDICT: confidence does NOT separate right from wrong. It is decoration,")
        print("           not a signal. Do not gate anything on it.")
    else:
        print("  VERDICT: confidence carries some signal.")

# ------------------------------------------------------- worst confusions ---
conf_pairs = Counter((e, p) for e, p, _, _, _ in pairs if e != p)
example = {}
for e, p, _, q, iid in pairs:
    if e != p and (e, p) not in example:
        example[(e, p)] = (iid, q)

print()
print("=" * 78)
print("TOP CONFUSIONS")
print("=" * 78)
top = conf_pairs.most_common(5)
for (e, p), count in top:
    iid, q = example[(e, p)]
    print(f"  expected {e:<14} -> predicted {p:<14} x{count}")
    print(f"      {iid}: {q[:88]}")

# the specific suspicion: comparison queries landing in multihop
comp_as_multi = conf_pairs.get(("compound", "multihop"), 0)
multi_as_comp = conf_pairs.get(("multihop", "compound"), 0)
print()
print(f"  compound predicted as multihop : {comp_as_multi}")
print(f"  multihop predicted as compound : {multi_as_comp}")

# ------------------------------------------------------------------ save ---
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "items_scored": n,
    "skipped": dict(skipped),
    "accuracy": round(accuracy, 4),
    "majority_class_baseline": round(majority[1] / n, 4),
    "majority_class": majority[0],
    "confusion_matrix": {e: dict(matrix.get(e, Counter())) for e in CLASSES},
    "per_class": per_class,
    "confidence_when_correct": sc,
    "confidence_when_wrong": sw,
    "top_confusions": [{"expected": e, "predicted": p, "count": c} for (e, p), c in top],
}, indent=2), encoding="utf-8")

print()
print("=" * 78)
print(f"Saved: {OUT}")
print("=" * 78)
print("Read the accuracy against the majority-class baseline, not against zero.")
print("Nothing here was tuned. Do not tune the classifier on this eval set.")
