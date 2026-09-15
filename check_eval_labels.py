"""Is `query_class_expected` coherent, or is the classifier being graded against noise?

The classifier scores 18.3% against a 20.1% majority baseline. That is either a
bad classifier or bad ground truth. This tells them apart without any hand
labelling: if near-identical queries carry different expected classes, the
labelling is inconsistent and no classifier could score well against it.

Reads only. Writes nothing.

    python check_eval_labels.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

EVAL = Path("data/eval/eval_set.jsonl")
STOP = set("""a an the of to in on for and or is are was were be been with as at by from
that this these those it its what which how why when who whom whose does do did can could
would should may might will shall between using used use their there here than then""".split())

if not EVAL.exists():
    sys.exit(f"Eval set not found at {EVAL}. Run from the repo root.")

rows = [json.loads(l) for l in EVAL.open(encoding="utf-8") if l.strip()]
items = [r for r in rows
         if r.get("provenance") != "synthetic_rejected"
         and r.get("query_class_expected")
         and (r.get("query") or "").strip()]

print("=" * 78)
print("EVAL LABEL COHERENCE")
print("=" * 78)
print(f"  labelled items: {len(items)}")
print(f"  class support : {dict(Counter(r['query_class_expected'] for r in items))}")


def content(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2]


# ------------------------------------------------ 1. shared opening phrase ---
print()
print("=" * 78)
print("1. DO IDENTICAL OPENINGS CARRY DIFFERENT LABELS?")
print("=" * 78)
print("  Grouping by the first 4 words of the query verbatim.")

groups = defaultdict(list)
for r in items:
    key = " ".join((r["query"] or "").lower().split()[:4])
    groups[key].append(r)

conflicted = 0
shown = 0
for key, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    if len(g) < 3:
        continue
    dist = Counter(r["query_class_expected"] for r in g)
    if len(dist) > 1:
        conflicted += len(g)
        if shown < 6:
            print(f"\n  \"{key}...\"  ({len(g)} queries)")
            for cls, n in dist.most_common():
                print(f"      {cls:<14} {n}")
            shown += 1

print(f"\n  items in an opening-group whose labels disagree: {conflicted}/{len(items)}"
      f"  ({conflicted / len(items):.1%})")

# --------------------------------------------- 2. near-duplicate vocabulary ---
print()
print("=" * 78)
print("2. DO NEAR-DUPLICATE QUERIES CARRY DIFFERENT LABELS?")
print("=" * 78)
print("  Jaccard overlap of content words > 0.55 = the same question, reworded.")

toks = {r["item_id"]: set(content(r["query"])) for r in items}
by_id = {r["item_id"]: r for r in items}
clashes = []
for a, b in combinations(items, 2):
    ta, tb = toks[a["item_id"]], toks[b["item_id"]]
    if not ta or not tb:
        continue
    j = len(ta & tb) / len(ta | tb)
    if j > 0.55 and a["query_class_expected"] != b["query_class_expected"]:
        clashes.append((j, a, b))

clashes.sort(key=lambda x: -x[0])
print(f"  near-duplicate pairs with DIFFERENT expected classes: {len(clashes)}")
for j, a, b in clashes[:4]:
    print(f"\n  overlap {j:.2f}")
    print(f"    [{a['query_class_expected']:<13}] {a['query'][:76]}")
    print(f"    [{b['query_class_expected']:<13}] {b['query'][:76]}")

# ----------------------------------------- 3. was the label assigned by rota? ---
print()
print("=" * 78)
print("3. WERE LABELS ASSIGNED MECHANICALLY?")
print("=" * 78)
print("  If the generator cycled through classes, the label sequence repeats.")

seq = [r["query_class_expected"] for r in sorted(items, key=lambda r: r.get("item_id", ""))]
print(f"  first 24 labels in item_id order:")
for i in range(0, min(24, len(seq)), 6):
    print("      " + ", ".join(seq[i:i + 6]))

best_p, best_score = None, 0.0
for period in range(2, 13):
    if len(seq) < period * 3:
        break
    matches = sum(1 for i in range(len(seq) - period) if seq[i] == seq[i + period])
    score = matches / (len(seq) - period)
    if score > best_score:
        best_p, best_score = period, score

print(f"\n  strongest repeat: period {best_p}, {best_score:.1%} of positions match")
if best_score > 0.5:
    print("  VERDICT: labels look assigned on a rota, not from the query's content.")
else:
    print("  VERDICT: no strong periodicity - labels are not a simple rota.")

# ----------------------------------------------------------- 4. verdict ---
print()
print("=" * 78)
print("READING THIS")
print("=" * 78)
print("  High conflict + many near-duplicate clashes  -> the LABELS are the problem.")
print("  The classifier is being graded against noise and 18.3% means nothing.")
print()
print("  Low conflict + few clashes                   -> the labels are coherent")
print("  and the classifier genuinely is at chance.")
