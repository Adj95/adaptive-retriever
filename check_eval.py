"""Inspect (and optionally clean) the eval set.

    python check_eval.py           # inspect, 5 samples
    python check_eval.py 12        # inspect, 12 samples
    python check_eval.py --fix     # drop items whose QUERY contains a forbidden
                                   # phrase, write eval_set.jsonl in place,
                                   # keep a .bak of the original
"""
import json
import shutil
import sys
from collections import Counter

PATH = "data/eval/eval_set.jsonl"

FIX = "--fix" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("-")]
N = int(args[0]) if args else 5

CLASS_FIELD = "query_class_expected"
PROV_FIELD = "provenance"

FORBIDDEN = [
    "the provided text", "the passage", "the excerpt", "this chunk",
    "according to the text", "in the text", "the document above",
    "here is a question", "the following text", "the text above",
    "mentioned in the text", "described in the text",
]

rows = []
with open(PATH, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"items: {len(rows)}\n")

print("query_class_expected")
print("-" * 56)
for cls, n in Counter(r.get(CLASS_FIELD, "MISSING") for r in rows).most_common():
    print(f"  {str(cls):<18}{n:>4}  {'#' * min(40, n)}")

print("\nprovenance")
print("-" * 56)
for st, n in Counter(r.get(PROV_FIELD, "MISSING") for r in rows).most_common():
    print(f"  {str(st):<18}{n:>4}")

print("\ndoc_class_of_source")
print("-" * 56)
for dc, n in Counter(r.get("doc_class_of_source", "MISSING") for r in rows).most_common():
    print(f"  {str(dc):<18}{n:>4}")

# --- forbidden phrases in the QUERY ---
flagged = []
for r in rows:
    q = (r.get("query") or "").lower()
    hits = [p for p in FORBIDDEN if p in q]
    if hits:
        flagged.append((r, hits))

print("\nforbidden-phrase scan")
print("-" * 56)
if flagged:
    print(f"  {len(flagged)} of {len(rows)} queries contain a forbidden phrase:\n")
    for r, hits in flagged[:12]:
        print(f"  [{r.get('item_id')}] {hits}")
        print(f"     {r.get('query','')[:150]}\n")
else:
    print("  clean")

# --- verbatim overlap between query and its passage ---
leaky = []
for r in rows:
    q = r.get("query") or ""
    ps = r.get("relevant_passages") or []
    ptext = (ps[0].get("passage_text", "") if ps else "") or ""
    qw = q.split()
    for i in range(len(qw) - 7):
        gram = " ".join(qw[i : i + 8])
        if gram and gram in ptext:
            leaky.append((r, gram))
            break

print("\nverbatim-overlap scan (8-gram from query found in its passage)")
print("-" * 56)
if leaky:
    print(f"  {len(leaky)} of {len(rows)} flagged. Judge these yourself during")
    print(f"  eval-verify - a shared proper noun is innocent, a copied clause is not.\n")
    for r, gram in leaky[:12]:
        print(f"  [{r.get('item_id')}] '{gram}'")
        print(f"     {r.get('query','')[:150]}\n")
else:
    print("  clean")

# --- priority review list for eval-verify ---
priority = {r.get("item_id") for r, _ in flagged} | {r.get("item_id") for r, _ in leaky}
if priority:
    print("\nreview these item_ids FIRST during eval-verify")
    print("-" * 56)
    print("  " + ", ".join(sorted(x for x in priority if x)))

# --- samples ---
print("\n" + "=" * 74)
print(f"{N} SAMPLES - READ THESE YOURSELF")
print("=" * 74)
for r in rows[:N]:
    ps = r.get("relevant_passages") or []
    ptext = (ps[0].get("passage_text", "") if ps else "(no passage)") or ""
    print(f"\n[{r.get(CLASS_FIELD,'?')}]  id={r.get('item_id','?')}  "
          f"src={r.get('doc_class_of_source','?')}  passages={len(ps)}  "
          f"prov={r.get(PROV_FIELD,'?')}")
    print(f"  Q: {r.get('query','')}")
    print(f"  P: {ptext[:300].replace(chr(10), ' ')}")

# --- optional cleanup ---
if FIX:
    if not flagged:
        print("\nnothing to fix.")
    else:
        drop = {r.get("item_id") for r, _ in flagged}
        kept = [r for r in rows if r.get("item_id") not in drop]
        shutil.copyfile(PATH, PATH + ".bak")
        with open(PATH, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nFIXED: dropped {len(drop)} items with forbidden phrases.")
        print(f"  {len(rows)} -> {len(kept)} items")
        print(f"  original saved as {PATH}.bak")
        print("  verbatim-overlap items were NOT dropped - judge those by hand.")
else:
    print("\n(run with --fix to drop the forbidden-phrase items)")
