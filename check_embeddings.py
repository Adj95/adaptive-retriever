"""Prove the embeddings are real before rebuilding anything.

Run from the project root with the venv active:

    python check_embeddings.py

PASS means similar sentences score high and different sentences score low.
FAIL means the vectors are still meaningless and you must NOT rebuild.
"""
import sys

try:
    from adaptive_retriever.vectorstore import DenseEmbeddingModel, SparseEmbeddingModel
except Exception as e:  # noqa: BLE001
    sys.exit(f"FAIL - could not import from vectorstore.py: {e}")


SIMILAR = (
    "The Federal Reserve raised interest rates to combat inflation.",
    "The Fed hiked rates to fight rising prices.",
)
DIFFERENT = (
    "The Federal Reserve raised interest rates to combat inflation.",
    "Photosynthesis converts light energy into chemical energy in plants.",
)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


print("Loading the dense embedding model (first run downloads ~130 MB)...")
dense = DenseEmbeddingModel()
vecs = dense.encode_batch([SIMILAR[0], SIMILAR[1], DIFFERENT[1]])
sim = cosine(vecs[0], vecs[1])
dif = cosine(vecs[0], vecs[2])

print()
print(f"  model      : {dense.model_name} on {dense.device}")
print(f"  dimensions : {len(vecs[0])}")
print()
print(f"  SIMILAR  sentences -> {sim:.4f}")
print(f"  DIFFERENT sentences -> {dif:.4f}")
print()

dense_ok = sim > 0.70 and sim > dif + 0.20
print("  DENSE:", "PASS" if dense_ok else "FAIL")
if not dense_ok:
    print("    Expected SIMILAR above 0.70 and DIFFERENT below 0.45.")
    print("    If both are near 0.00-0.03 the old hash vectors are still in use.")

print()
print("Loading the sparse BM25 model (first run downloads a small file)...")
sparse = SparseEmbeddingModel()
s1, s2 = sparse.encode_batch([SIMILAR[0], DIFFERENT[1]])
shared = set(s1.indices) & set(s2.indices)

print()
print(f"  model            : {sparse.model_name}")
print(f"  terms sentence 1 : {len(s1.indices)}")
print(f"  terms sentence 2 : {len(s2.indices)}")
print(f"  shared terms     : {len(shared)}")

# Stability: same text must give identical indices on a second call.
again = sparse.encode_batch([SIMILAR[0]])[0]
stable = list(again.indices) == list(s1.indices)
print(f"  repeatable       : {stable}")

sparse_ok = len(s1.indices) > 0 and stable
print()
print("  SPARSE:", "PASS" if sparse_ok else "FAIL")
if not stable:
    print("    Indices changed between calls - vectors are not reproducible.")

print()
print("=" * 60)
if dense_ok and sparse_ok:
    print("ALL PASS - the embeddings are real. Safe to rebuild the index.")
    sys.exit(0)
print("FAILED - do NOT rebuild. Send this output back.")
sys.exit(1)
