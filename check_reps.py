"""Read real generated representations out of Qdrant and prove they are not templates.

Finds the Qdrant store itself - no hardcoded paths.

Run from the project root, venv active, and ONLY when no pipeline command is
running (Qdrant local mode holds a file lock):

    python check_reps.py                 # auto-pick the biggest retriever_chunks_* collection
    python check_reps.py a94fcb78        # pick by hash fragment
"""
import os
import sys
from collections import Counter

try:
    from qdrant_client import QdrantClient
except ImportError:
    sys.exit("qdrant_client not importable - is the venv active?")

TEMPLATE_MARKERS = (
    "Summary of chunk",
    "This chunk discusses",
    "The following text",
    "What specific value",
    "is specified for",
)

TEXT_FIELDS = ("text", "content", "representation_text", "page_content", "chunk_text")
KIND_FIELDS = ("kind", "representation_kind", "rep_kind", "type")


def find_stores(root="."):
    """A qdrant local store is a dir containing meta.json and a collection/ subdir."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        parts = set(dirpath.split(os.sep))
        if parts & {".git", ".venv", "node_modules", "__pycache__"}:
            dirnames[:] = []
            continue
        if "meta.json" in filenames and "collection" in dirnames:
            try:
                n = len(os.listdir(os.path.join(dirpath, "collection")))
            except OSError:
                n = 0
            hits.append((dirpath, n))
            dirnames[:] = []
    return sorted(hits, key=lambda t: -t[1])


def get_text(payload):
    for f in TEXT_FIELDS:
        if payload.get(f):
            return f, str(payload[f])
    return None, ""


def get_kind(payload):
    for f in KIND_FIELDS:
        if f in payload:
            return str(payload[f])
    return "UNKNOWN"


stores = find_stores(".")
if not stores:
    sys.exit("No Qdrant store found under the project root.")

print("qdrant stores found (path, collections):")
for p, n in stores:
    print(f"  {p}   -> {n} collection dir(s)")
store_path = stores[0][0]
print(f"\nusing: {store_path}\n")

client = QdrantClient(path=store_path)
try:
    cols = [c.name for c in client.get_collections().collections]
    counts = {}
    for n in cols:
        try:
            counts[n] = client.count(n, exact=True).count
        except Exception as exc:  # noqa: BLE001
            counts[n] = f"error: {exc}"

    print("collections present:")
    for n in cols:
        print(f"  {n:<36} {counts[n]}")

    if not cols:
        sys.exit("\nStore has no collections. Wrong store - check the paths listed above.")

    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    if wanted:
        matches = [n for n in cols if wanted in n]
        if not matches:
            sys.exit(f"\nNo collection matching '{wanted}'.")
        target = matches[0]
    else:
        numeric = {n: c for n, c in counts.items()
                   if isinstance(c, int) and n.startswith("retriever_chunks")}
        if not numeric:
            sys.exit("\nNo retriever_chunks_* collection. Pass a name fragment as an argument.")
        target = max(numeric, key=numeric.get)

    print(f"\ninspecting: {target}  ({counts[target]} points)\n")

    kinds = Counter()
    samples = {}
    suspect = Counter()
    offset = None
    scanned = 0

    while scanned < 6000:
        pts, offset = client.scroll(
            target, limit=500, offset=offset, with_payload=True, with_vectors=False
        )
        if not pts:
            break
        for p in pts:
            pl = p.payload or {}
            k = get_kind(pl)
            kinds[k] += 1
            _, txt = get_text(pl)
            if any(m in txt for m in TEMPLATE_MARKERS):
                suspect[k] += 1
            if k not in samples and txt:
                samples[k] = pl
        scanned += len(pts)
        if offset is None:
            break

    print(f"scanned {scanned} points")
    print("kinds seen:", dict(kinds))
    print("TEMPLATE-LOOKING:", dict(suspect) if suspect else "none")
    print()

    for k, pl in sorted(samples.items()):
        field, txt = get_text(pl)
        print("=" * 72)
        print(f"KIND: {k}    (text field: {field})")
        print(f"payload keys: {sorted(pl.keys())}")
        print("-" * 72)
        print(txt[:800])
        print()

    print("=" * 72)
    print("READ THE summary AND question BLOCKS ABOVE YOURSELF.")
    print("Real summary  = paraphrases the chunk in different words.")
    print("Template      = restates it, or 'Summary of chunk N:' + raw text.")
finally:
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass
