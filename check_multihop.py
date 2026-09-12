"""Check whether multihop eval items pair a real passage with a random one.

    python check_multihop.py
"""
import json
import re
from collections import Counter

PATH = "data/eval/eval_set.jsonl"
STOP = set("""a an the of to in on for and or is are was were be been with as at by from
that this these those it its what which how why when who whom whose does do did can could
would should may might will shall between using used use their there here than then""".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 2}


rows = [json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()]
multi = [r for r in rows if r.get("query_class_expected") == "multihop"]

print(f"multihop items: {len(multi)} of {len(rows)}\n")

bad = 0
counts = Counter()
print(f"{'item_id':<14}{'#p':>3}{'p1 overlap':>12}{'p2 overlap':>12}   verdict")
print("-" * 66)

for r in multi:
    q = toks(r.get("query"))
    ps = r.get("relevant_passages") or []
    counts[len(ps)] += 1
    if not q:
        continue
    ov = []
    for p in ps:
        pt = toks(p.get("passage_text"))
        ov.append(len(q & pt) / len(q) if q else 0.0)
    if len(ov) < 2:
        print(f"{r['item_id']:<14}{len(ps):>3}{ov[0] if ov else 0:>12.2f}{'-':>12}   ONLY 1 PASSAGE")
        bad += 1
        continue
    v = "OK" if min(ov[0], ov[1]) >= 0.12 else "PASSAGE 2 UNRELATED"
    if v != "OK":
        bad += 1
    print(f"{r['item_id']:<14}{len(ps):>3}{ov[0]:>12.2f}{ov[1]:>12.2f}   {v}")

print("-" * 66)
print(f"passage-count distribution: {dict(counts)}")
print(f"broken: {bad} / {len(multi)}")
print()
print("'overlap' = fraction of the query's content words that appear in that passage.")
print("A genuine multihop item needs BOTH passages to answer it, so both should")
print("share meaningful vocabulary with the query. A near-zero second column means")
print("the passage was paired at random and the item is unanswerable as written.")
