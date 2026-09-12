"""Diagnose the manifest <-> canonical parse join.

Run from the project root:
    python check_join.py
"""
import json
import hashlib
import os
from collections import defaultdict

MANIFEST = "data/corpus/MANIFEST.jsonl"
CANON_DIR = "data/canonical"

rows = []
with open(MANIFEST, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("_meta"):
            rows.append(r)

found = defaultdict(int)
missing = defaultdict(int)
no_file = defaultdict(int)
examples = []

for r in rows:
    path = r["doc_path"]
    cls = r.get("expected_doc_class", "UNKNOWN")

    if not os.path.exists(path):
        no_file[cls] += 1
        if len(examples) < 12:
            examples.append((cls, "SOURCE FILE MISSING", path))
        continue

    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]

    canon = os.path.join(CANON_DIR, digest + ".txt")
    if os.path.exists(canon):
        found[cls] += 1
    else:
        missing[cls] += 1
        if len(examples) < 12:
            examples.append((cls, "NO CANONICAL " + digest, path))

classes = sorted(set(list(found) + list(missing) + list(no_file)))

print()
print(f"{'class':<16}{'matched':>9}{'unmatched':>11}{'src gone':>10}")
print("-" * 46)
for cls in classes:
    print(f"{cls:<16}{found[cls]:>9}{missing[cls]:>11}{no_file[cls]:>10}")
print("-" * 46)
print(f"{'TOTAL':<16}{sum(found.values()):>9}{sum(missing.values()):>11}{sum(no_file.values()):>10}")

on_disk = [f for f in os.listdir(CANON_DIR) if f.endswith(".txt")]
print()
print(f"manifest rows          : {len(rows)}")
print(f"canonical files on disk: {len(on_disk)}")
print(f"matched by sha256[:16] : {sum(found.values())}")

if examples:
    print()
    print("first failures:")
    for cls, why, path in examples:
        print(f"  [{cls}] {why}")
        print(f"      {path}")
print()
