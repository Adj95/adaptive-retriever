"""Raw-Only Index Rebuild Script (TASK R-01a).

1. Re-chunks all 428 corpus documents using patched tokenizer-capped chunk_recursive_512.
2. Asserts no chunk exceeds 512 BGE-small tokens (FAILS the build on violation).
3. Embeds RAW CHUNKS ONLY into Qdrant (kind='raw', batch_size=64 on CUDA).
4. Reports chunk length distribution in CHARACTERS and TOKENS (min, p50, p95, p99, max).
"""

import time
import json
import logging
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

from adaptive_retriever.config import Config
from adaptive_retriever.ingest.parsers import parse_document_cached
from adaptive_retriever.ingest.chunkers import chunk_recursive_512
from adaptive_retriever.models import RetrievalCandidate, Chunk
from adaptive_retriever.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    t_start = time.perf_counter()
    cfg = Config.load("config/default.yaml")
    
    manifest_path = Path("data/corpus/MANIFEST.jsonl")
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    print(f"Loaded manifest with {len(records)} document records.")
    
    # Re-chunk all documents
    all_chunks = []
    t0_chunk = time.perf_counter()
    for idx, rec in enumerate(records, 1):
        fpath = Path(rec.get("doc_path", rec.get("file_path", "")))
        doc_id = rec["doc_id"]
        parsed, _, _ = parse_document_cached(fpath, parser_name="pymupdf", doc_id=doc_id)
        doc_chunks = chunk_recursive_512(parsed, doc_id=doc_id)
        all_chunks.extend(doc_chunks)
        if idx % 50 == 0 or idx == len(records):
            print(f"  Chunked {idx}/{len(records)} documents ({len(all_chunks)} chunks so far)...", flush=True)
    t1_chunk = time.perf_counter()
    chunk_time_s = t1_chunk - t0_chunk
    
    print(f"Re-chunked {len(records)} documents into {len(all_chunks)} RAW chunks in {chunk_time_s:.2f} s.")
    
    # Tokenizer verification and distribution calculation
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    
    char_lens = [len(c.text) for c in all_chunks]
    token_counts = []
    
    print("Tokenizing and verifying all chunks against 512 token cap...")
    for c in all_chunks:
        t_cnt = len(tok.encode(c.text, add_special_tokens=False))
        c.token_count = t_cnt
        token_counts.append(t_cnt)
        assert t_cnt <= 512, f"PERMANENT ASSERTION FAILED: Chunk {c.chunk_id} has {t_cnt} tokens > 512!"
        
    print(f"SUCCESS: Token count assertion passed for all {len(all_chunks)} chunks.")
    
    # Length distributions
    print("\n" + "="*80)
    print("CHUNK LENGTH DISTRIBUTION — CHARACTERS")
    print("="*80)
    print(f"Min: {np.min(char_lens):,}")
    print(f"p50: {np.percentile(char_lens, 50):,.1f}")
    print(f"p95: {np.percentile(char_lens, 95):,.1f}")
    print(f"p99: {np.percentile(char_lens, 99):,.1f}")
    print(f"Max: {np.max(char_lens):,}")

    print("\n" + "="*80)
    print("CHUNK LENGTH DISTRIBUTION — TOKENS (BGE-small)")
    print("="*80)
    print(f"Min: {np.min(token_counts)}")
    print(f"p50: {np.percentile(token_counts, 50):.1f}")
    print(f"p95: {np.percentile(token_counts, 95):.1f}")
    print(f"p99: {np.percentile(token_counts, 99):.1f}")
    print(f"Max: {np.max(token_counts)}")
    
    # Ingest into Qdrant server
    vs = VectorStore.from_config(cfg)
    print(f"\nIngesting {len(all_chunks)} RAW chunks into Qdrant collection '{vs.collection_name}'...")
    
    # Ensure collection exists and is fresh
    try:
        vs.client.delete_collection(vs.collection_name)
    except Exception:
        pass
    vs._ensure_collection()
    
    t0_embed = time.perf_counter()
    
    # Batch indexing RAW chunks ONLY
    batch_size = 64
    total_points = 0
    from qdrant_client.http import models as rest
    from adaptive_retriever.vectorstore import stable_point_id
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        
        # Dense embeddings (batched CUDA)
        dense_vecs = vs.embedding_client.encode_batch(texts)
        
        # Sparse embeddings
        sparse_vecs = vs.sparse_client.encode_batch(texts)
        
        points = []
        for c, d_vec, s_vec in zip(batch, dense_vecs, sparse_vecs):
            pt_id = f"{c.chunk_id}_raw"
            payload = {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "ordinal": c.ordinal,
                "kind": "raw",
                "rep_id": pt_id,
                "text": c.text,
                "token_count": c.token_count,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "chunker_used": c.chunker_used,
            }
            points.append(
                rest.PointStruct(
                    id=stable_point_id(pt_id),
                    vector={"dense": d_vec, "sparse": s_vec},
                    payload=payload,
                )
            )
            
        vs.client.upsert(collection_name=vs.collection_name, points=points)
        total_points += len(points)
        if (i // batch_size + 1) % 50 == 0 or (i + batch_size) >= len(all_chunks):
            print(f"  Indexed {total_points}/{len(all_chunks)} points...", flush=True)
            
    t1_embed = time.perf_counter()
    embed_time_s = t1_embed - t0_embed
    
    print(f"Indexed {total_points} RAW points in {embed_time_s:.2f} s ({total_points/embed_time_s:.1f} points/sec).")
    
    t_end = time.perf_counter()
    total_wall_clock_s = t_end - t_start
    
    summary = {
        "ingest_config_hash": cfg.ingest_config_hash,
        "collection_name": vs.collection_name,
        "total_documents": len(records),
        "total_chunks": len(all_chunks),
        "total_points": total_points,
        "max_tokens_asserted": max(token_counts),
        "token_distribution": {
            "min": int(np.min(token_counts)),
            "p50": float(np.percentile(token_counts, 50)),
            "p95": float(np.percentile(token_counts, 95)),
            "p99": float(np.percentile(token_counts, 99)),
            "max": int(np.max(token_counts)),
        },
        "char_distribution": {
            "min": int(np.min(char_lens)),
            "p50": float(np.percentile(char_lens, 50)),
            "p95": float(np.percentile(char_lens, 95)),
            "p99": float(np.percentile(char_lens, 99)),
            "max": int(np.max(char_lens)),
        },
        "wall_clock_seconds": round(total_wall_clock_s, 2),
    }
    
    with open("reports/r01a_rebuild_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nRebuild complete. Summary saved to reports/r01a_rebuild_stats.json")

if __name__ == "__main__":
    main()
