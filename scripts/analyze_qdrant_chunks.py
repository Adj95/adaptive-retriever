"""Analyze Qdrant index chunk length distribution and un-chunked document blocks (Task C4)."""

import json
import logging
import numpy as np
from pathlib import Path

from adaptive_retriever.config import Config
from adaptive_retriever.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    cfg = Config.load("config/default.yaml")
    vs = VectorStore.from_config(cfg)
    
    print(f"Connecting to Qdrant collection: '{vs.collection_name}' (mode: {vs.mode})")
    
    # Scroll all points in Qdrant collection
    all_points = []
    offset = None
    while True:
        res, offset = vs.client.scroll(
            collection_name=vs.collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(res)
        if offset is None:
            break
            
    print(f"Total points retrieved from Qdrant: {len(all_points)}")
    
    # Extract chunk texts and lengths
    # Group by chunk_id or inspect all payloads
    chunk_lengths = []
    chunk_details = []
    
    seen_chunks = set()
    for pt in all_points:
        p = pt.payload or {}
        chunk_id = p.get("chunk_id") or str(pt.id)
        raw_text = p.get("text", "")
        doc_id = p.get("doc_id", "unknown")
        
        # Avoid duplicate counting of representations for the same chunk
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        
        length = len(raw_text)
        chunk_lengths.append(length)
        chunk_details.append({
            "point_id": pt.id,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "length": length,
            "text_snippet": raw_text[:150] + "..." if len(raw_text) > 150 else raw_text,
            "kind": p.get("kind"),
            "doc_class": p.get("doc_class"),
        })
        
    arr = np.array(chunk_lengths)
    print("\n=== RAW CHUNK TEXT LENGTH DISTRIBUTION (CHARS) ===")
    print(f"Total Unique Chunks: {len(arr)}")
    print(f"Min: {np.min(arr):,}")
    print(f"p50 (Median): {np.percentile(arr, 50):,.1f}")
    print(f"p95: {np.percentile(arr, 95):,.1f}")
    print(f"p99: {np.percentile(arr, 99):,.1f}")
    print(f"Max: {np.max(arr):,}")
    
    over_2k = np.sum(arr > 2000)
    over_10k = np.sum(arr > 10000)
    over_50k = np.sum(arr > 50000)
    
    print("\n=== OVERSIZED CHUNK COUNTS ===")
    print(f"Chunks > 2,000 chars:  {over_2k:,} ({over_2k/len(arr)*100:.2f}%)")
    print(f"Chunks > 10,000 chars: {over_10k:,} ({over_10k/len(arr)*100:.2f}%)")
    print(f"Chunks > 50,000 chars: {over_50k:,} ({over_50k/len(arr)*100:.2f}%)")
    
    # Sort by length descending to get largest 5
    sorted_chunks = sorted(chunk_details, key=lambda x: x["length"], reverse=True)
    largest_5 = sorted_chunks[:5]
    
    print("\n=== LARGEST 5 CHUNKS ===")
    for idx, c in enumerate(largest_5, 1):
        print(f"\n{idx}. Chunk ID: {c['chunk_id']} | Doc ID: {c['doc_id']} | Length: {c['length']:,} chars")
        print(f"   Class: {c['doc_class']} | Kind: {c['kind']}")
        print(f"   Snippet: {c['text_snippet']!r}")

if __name__ == "__main__":
    main()
