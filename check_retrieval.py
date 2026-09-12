"""Prove the index is searchable, before building the Phase 2 funnel on top of it.

Calls VectorStore.query_hybrid directly: embed the query -> dense + sparse
prefetch -> RRF fusion. No CLI, no funnel.

    python check_retrieval.py
    python check_retrieval.py "your own question here"
"""
import sys

from adaptive_retriever.config import Config
from adaptive_retriever.vectorstore import VectorStore

QUERIES = [
    "What did the Federal Reserve say about inflation and interest rates?",
    "How does the attention mechanism work in transformer models?",
    "What are the risks disclosed in the annual report?",
]

if len(sys.argv) > 1:
    QUERIES = [" ".join(sys.argv[1:])]

cfg = Config.load("config/default.yaml")
store = VectorStore.from_config(cfg)

print(f"collection : {store.collection_name}")
print(f"points     : {store.count()}")

try:
    for q in QUERIES:
        print("\n" + "=" * 74)
        print(f"QUERY: {q}")
        print("=" * 74)
        results = store.query_hybrid(q, limit=5)
        if not results:
            print("  NO RESULTS - something is wrong.")
            continue
        for r in results:
            md = r.metadata or {}
            text = (r.text or "").replace("\n", " ")
            print(f"\n  #{r.final_rank}  score={r.fused_score:.4f}  "
                  f"kind={md.get('kind')}  class={md.get('doc_class')}")
            print(f"      {text[:260]}")
finally:
    store.close()

print("\n" + "=" * 74)
print("READ THE RESULTS YOURSELF.")
print("Good  = the text is actually about what you asked.")
print("Bad   = unrelated text, or every score nearly identical.")
