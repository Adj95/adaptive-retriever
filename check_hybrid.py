"""End-to-end check of the Phase 2 hybrid funnel against the LIVE index.

The unit tests prove the logic with mocks. This proves it against the real
22,422-point collection, which is where the interesting failures live.

Checks:
  1. every query class in models.QueryClass runs without raising
  2. no chunk appears twice in one result set
  3. the text returned really is the chunk's RAW text, not a summary
  4. which representation kind is winning retrieval (the Phase 6 headline)
  5. class weights actually change the ranking

Needs the Qdrant container running. Reads only, writes nothing.

    python check_hybrid.py
    python check_hybrid.py --query "your own question here"
"""
import argparse
import sys
from collections import Counter

from adaptive_retriever.config import Config
from adaptive_retriever.models import QueryClass
from adaptive_retriever.vectorstore import VectorStore, stable_point_id

ap = argparse.ArgumentParser()
ap.add_argument("--query", default="what methods were used to evaluate the results")
ap.add_argument("--limit", type=int, default=10)
args = ap.parse_args()

cfg = Config.load("config/default.yaml")
vs = VectorStore.from_config(cfg)

print("=" * 72)
print("SETUP")
print("=" * 72)
print(f"  collection : {vs.collection_name}")
print(f"  query      : {args.query}")
print(f"  limit      : {args.limit}")

failures = []

# ---------------------------------------------------- 1. every class runs ---
print()
print("=" * 72)
print("1. EVERY QUERY CLASS RUNS")
print("=" * 72)
results_by_class = {}
for qc in QueryClass:
    try:
        cands = vs.query_hybrid(args.query, limit=args.limit, query_class=qc.value)
        results_by_class[qc.value] = cands
        w = cfg.raw["funnel"]["hybrid"]["class_weights"][qc.value]
        print(f"  {qc.value:<14} OK   {len(cands)} candidates   weights={w}")
    except Exception as e:  # noqa: BLE001
        print(f"  {qc.value:<14} RAISED  {type(e).__name__}: {e}")
        failures.append(f"query_class '{qc.value}' raised")

if not results_by_class:
    print("\n  Nothing ran. Is the Qdrant container up?  docker ps")
    sys.exit(1)

# ------------------------------------------------------- 2. no duplicates ---
print()
print("=" * 72)
print("2. NO CHUNK APPEARS TWICE")
print("=" * 72)
for cls, cands in results_by_class.items():
    ids = [c.chunk_id for c in cands]
    dupes = [cid for cid, n in Counter(ids).items() if n > 1]
    if dupes:
        print(f"  {cls:<14} FAIL  repeated chunk_ids: {dupes[:3]}")
        failures.append(f"duplicate chunks for '{cls}'")
    else:
        print(f"  {cls:<14} OK    {len(set(ids))} distinct chunks")

# ------------------------------------------------- 3. text is the RAW text ---
print()
print("=" * 72)
print("3. RETURNED TEXT IS THE CHUNK'S RAW TEXT")
print("=" * 72)
print("  (a summary here would mean the generator cites model-written text)")
sample = results_by_class.get("conceptual") or next(iter(results_by_class.values()))
checked = mismatched = 0
for c in sample[:5]:
    pts = vs.client.retrieve(
        collection_name=vs.collection_name,
        ids=[stable_point_id(f"{c.chunk_id}_raw")],
        with_payload=True,
        with_vectors=False,
    )
    if not pts:
        print(f"  {c.chunk_id[:34]:<36} NO RAW POINT")
        failures.append(f"no raw point for {c.chunk_id}")
        continue
    raw_text = (pts[0].payload or {}).get("text", "")
    checked += 1
    same = raw_text == c.text
    if not same:
        mismatched += 1
    mk = c.metadata.get("matched_kind", "?")
    print(f"  {c.chunk_id[:34]:<36} matched_by={mk:<9} text==raw: {same}")

if mismatched:
    print(f"\n  FAIL  {mismatched}/{checked} returned a representation's text, not the chunk's")
    failures.append("returned text is not the raw chunk text")
elif checked:
    print(f"\n  PASS  all {checked} checked candidates carry the chunk's own text")

# --------------------------------------------- 4. which kind is winning ----
print()
print("=" * 72)
print("4. WHICH REPRESENTATION KIND WINS RETRIEVAL")
print("=" * 72)
print("  This is a Phase 6 result in its own right: it says whether generating")
print("  summaries and questions at ingest actually buys retrieval quality.")
for cls, cands in results_by_class.items():
    kinds = Counter(c.metadata.get("matched_kind", "?") for c in cands)
    line = "  ".join(f"{k}={n}" for k, n in sorted(kinds.items()))
    print(f"  {cls:<14} {line}")

# ------------------------------------------------ 5. weights change things --
print()
print("=" * 72)
print("5. CLASS WEIGHTS CHANGE THE RANKING")
print("=" * 72)
a = [c.chunk_id for c in results_by_class.get("exact_lookup", [])]
b = [c.chunk_id for c in results_by_class.get("conceptual", [])]
if not a or not b:
    print("  SKIP  one of the two classes returned nothing")
elif a == b:
    print("  WARN  exact_lookup (0.3/0.7) and conceptual (0.8/0.2) gave the")
    print("        IDENTICAL ranking. Either this query is insensitive to the")
    print("        weights, or the weights are not reaching fusion.")
    print("        Re-run with --query on something more lexical before worrying.")
else:
    overlap = len(set(a) & set(b))
    print(f"  PASS  rankings differ  (top-{len(a)} overlap: {overlap})")
    print(f"        exact_lookup top 3: {a[:3]}")
    print(f"        conceptual   top 3: {b[:3]}")

# ------------------------------------------------------------- verdict -----
print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
if failures:
    for f in failures:
        print(f"  FAIL  {f}")
    sys.exit(1)
print("  PASS  hybrid funnel behaves correctly against the live index")
sys.exit(0)
