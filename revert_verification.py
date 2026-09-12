"""Revert fabricated verification marks on the eval set.

`cli eval-verify` stamped provenance='synthetic_verified' on 50 items without
showing a single one to a human. Nothing in this project may claim verification
that did not happen.

    python revert_verification.py           # show what would change
    python revert_verification.py --apply   # write the change (keeps a .bak)
"""
import json
import shutil
import sys
from collections import Counter

PATH = "data/eval/eval_set.jsonl"
APPLY = "--apply" in sys.argv

rows = []
with open(PATH, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

before = Counter(r.get("provenance") for r in rows)
print(f"items: {len(rows)}")
print("provenance now :", dict(before))

marked = [r for r in rows if r.get("provenance") == "synthetic_verified"]
if not marked:
    sys.exit("\nNothing marked verified. Nothing to revert.")

print(f"\n{len(marked)} items marked 'synthetic_verified'")
print(f"  range: {marked[0]['item_id']} -> {marked[-1]['item_id']}")

ids = [r["item_id"] for r in marked]
seq = ids == [f"eval_{i:05d}" for i in range(1, len(ids) + 1)]
print(f"  contiguous from the first item: {seq}")
if seq:
    print("  -> that is rows[:50], not a human review. Confirms auto-marking.")

missing_ts = sum(1 for r in marked if not r.get("verified_at"))
print(f"  without a verified_at timestamp: {missing_ts}/{len(marked)}")

if not APPLY:
    print("\nDry run. Re-run with --apply to revert these to 'synthetic'.")
    sys.exit(0)

shutil.copyfile(PATH, PATH + ".preverify.bak")
for r in rows:
    if r.get("provenance") == "synthetic_verified":
        r["provenance"] = "synthetic"
        r.pop("verified_at", None)
        r.pop("verified_by", None)

with open(PATH, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

after = Counter(r.get("provenance") for r in rows)
print(f"\nREVERTED. provenance now: {dict(after)}")
print(f"  backup of the bad state: {PATH}.preverify.bak")
print("\nNo item in this eval set now claims verification that did not happen.")
