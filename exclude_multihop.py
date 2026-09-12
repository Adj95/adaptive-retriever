"""Exclude multihop eval items whose second passage is unrelated to the query.

These items are UNANSWERABLE as written: a genuine multihop question needs both
passages, but the generator paired a relevant passage with a randomly sampled
one. Retrieval can never return the random passage, so the item would silently
depress Recall.

This marks them provenance='synthetic_rejected' with rejected_by='automated_criterion'
so the record is honest about how they were excluded. It does NOT mark anything
as verified.

    python exclude_multihop.py            # dry run
    python exclude_multihop.py --apply    # write (keeps a .bak)
"""
import json
import re
import shutil
import sys
from collections import Counter

PATH = "data/eval/eval_set.jsonl"
APPLY = "--apply" in sys.argv
THRESHOLD = 0.12

STOP = set("""a an the of to in on for and or is are was were be been with as at by from
that this these those it its what which how why when who whom whose does do did can could
would should may might will shall between using used use their there here than then""".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 2}


rows = [json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()]
print(f"items: {len(rows)}")
print("provenance now:", dict(Counter(r.get("provenance") for r in rows)))

to_exclude = []
for r in rows:
    if r.get("query_class_expected") != "multihop":
        continue
    if r.get("provenance") in ("synthetic_verified", "synthetic_rejected"):
        continue  # a human already decided - leave it alone
    q = toks(r.get("query"))
    ps = r.get("relevant_passages") or []
    if not q:
        continue
    ov = [len(q & toks(p.get("passage_text"))) / len(q) for p in ps]
    if len(ov) < 2 or min(ov) < THRESHOLD:
        to_exclude.append((r, round(min(ov) if ov else 0.0, 3)))

print(f"\nmultihop items to exclude: {len(to_exclude)}")
for r, score in to_exclude[:8]:
    print(f"  {r['item_id']}  worst-passage overlap {score:.2f}")
if len(to_exclude) > 8:
    print(f"  ... and {len(to_exclude) - 8} more")

if not APPLY:
    print("\nDry run. Re-run with --apply to exclude these.")
    sys.exit(0)

shutil.copyfile(PATH, PATH + ".premultihop.bak")
for r, score in to_exclude:
    r["provenance"] = "synthetic_rejected"
    r["rejected_by"] = "automated_criterion"
    r["rejected_reason"] = (
        f"multihop item: a relevant passage was paired with an unrelated one "
        f"(lowest query/passage content-word overlap {score:.2f} < {THRESHOLD}). "
        f"Unanswerable as written; excluded from reported metrics."
    )

with open(PATH, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nEXCLUDED {len(to_exclude)} items.")
print("provenance now:", dict(Counter(r.get("provenance") for r in rows)))
print(f"backup: {PATH}.premultihop.bak")
print("\nNothing was marked verified. Excluded items carry rejected_by='automated_criterion'.")
