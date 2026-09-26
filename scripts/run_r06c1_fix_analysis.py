"""TASK R-06c-1-fix: Part 1 Analysis & Report Correction.

1. Analysis of 10 HyDE hypothetical documents vs raw queries & gold docs.
2. Cosine similarity comparison (HyDE vs Gold vs Raw Query vs Gold).
3. Relabeling T3, updating multiplicity (9 tests, alpha = 0.00556), and updating reports/r06c1_transformations.json.
"""

import json
import math
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")

from adaptive_retriever.config import Config
from adaptive_retriever.vectorstore import VectorStore
from adaptive_retriever.benchmarks.beir import extract_doc_id
from qdrant_client.http import models as rest

def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def load_qrels_tsv(path: Path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        header = next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, docid, score = parts[0], parts[1], int(parts[2])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = score
    return qrels

def main():
    print("=== PART 1: R-06c-1 FIX ANALYSIS ===")
    
    # Load cache
    cache_path = Path("data/beir/nfcorpus/transformations_cache.json")
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    queries_path = Path("data/beir/nfcorpus/queries.jsonl")
    qrels_path = Path("data/beir/nfcorpus/qrels/test.tsv")
    
    qrels = load_qrels_tsv(qrels_path)
    all_queries = load_jsonl(queries_path)
    test_queries = [q for q in all_queries if str(q["_id"]) in qrels]
    
    hyde_passages = cache["hyde_passages"]
    
    # 1. Print 10 verbatim query + HyDE passage pairs
    print("\n--- 10 VERBATIM HYDE PASSAGE SAMPLES ---")
    for idx in range(10):
        q = test_queries[idx]
        hp = hyde_passages[idx]
        print(f"\n[Query {idx+1} | ID: {q['_id']}] {q['text']}")
        print(f"[HyDE Generated Passage]:\n{hp}\n" + "-"*50)
        
    # 2. Compute token lengths across all 323 HyDE passages
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    
    tok_lens = [len(tokenizer.encode(hp, add_special_tokens=False)) for hp in hyde_passages]
    mean_tok = float(np.mean(tok_lens))
    p50_tok = float(np.median(tok_lens))
    print(f"\nHyDE Passage Token Lengths: mean={mean_tok:.1f}, p50={p50_tok:.1f}")
    
    # 3. Quality breakdown assessment
    coherent_cnt = 0
    generic_cnt = 0
    degenerate_cnt = 0
    
    for hp in hyde_passages:
        hp_lower = hp.lower()
        if len(hp.strip()) < 10 or "i cannot" in hp_lower or "as an ai" in hp_lower or hp.count(hp.split()[0] if hp.split() else "") > 10:
            degenerate_cnt += 1
        elif any(k in hp_lower for k in ["study", "patient", "clinical", "treatment", "effect", "disease", "risk", "intake", "diet", "cancer", "health", "supplement", "blood", "heart", "food"]):
            coherent_cnt += 1
        else:
            generic_cnt += 1
            
    print(f"Quality Breakdown: Coherent Clinical={coherent_cnt} ({coherent_cnt/323*100:.1f}%), Generic Filler={generic_cnt} ({generic_cnt/323*100:.1f}%), Degenerate={degenerate_cnt} ({degenerate_cnt/323*100:.1f}%)")

    # 4. Cosine Similarity Analysis: (HyDE vs Gold Docs) vs (Raw Query vs Gold Docs)
    cfg = Config.load("config/beir_nfcorpus.yaml")
    vs = VectorStore.from_config(cfg, collection_base_name="beir_nfcorpus")
    
    print("\nComputing embedding cosines between (HyDE, Raw Query) and Gold Docs across test queries...")
    hyde_gold_cosines = []
    query_gold_cosines = []
    
    for idx, q in enumerate(test_queries[:150]): # sample 150 test queries
        qid = str(q["_id"])
        raw_text = q["text"]
        hp_text = hyde_passages[idx]
        
        gold_doc_ids = [did for did, score in qrels[qid].items() if score >= 1]
        if not gold_doc_ids:
            continue
            
        q_vec = np.array(vs.embedding_client.encode(raw_text))
        hp_vec = np.array(vs.embedding_client.encode(hp_text))
        
        # Scroll points for first gold doc
        target_doc_id = gold_doc_ids[0]
        flt = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="doc_id",
                    match=rest.MatchValue(value=target_doc_id)
                )
            ]
        )
        res, _ = vs.client.scroll(
            collection_name=vs.collection_name,
            scroll_filter=flt,
            with_vectors=True,
            limit=1,
        )
        if res and res[0].vector:
            g_vec = res[0].vector.get("dense") if isinstance(res[0].vector, dict) else res[0].vector
            if g_vec is not None:
                g_vec = np.array(g_vec)
                q_vec_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
                hp_vec_norm = hp_vec / (np.linalg.norm(hp_vec) + 1e-9)
                g_vec_norm = g_vec / (np.linalg.norm(g_vec) + 1e-9)
                
                cos_q_g = float(np.dot(q_vec_norm, g_vec_norm))
                cos_hp_g = float(np.dot(hp_vec_norm, g_vec_norm))
                
                query_gold_cosines.append(cos_q_g)
                hyde_gold_cosines.append(cos_hp_g)
                
    mean_cos_q_g = float(np.mean(query_gold_cosines))
    mean_cos_hp_g = float(np.mean(hyde_gold_cosines))
    print(f"\nMean Cosine Similarity to Gold Documents (n={len(query_gold_cosines)}):")
    print(f"  Raw Query <-> Gold Docs   : {mean_cos_q_g:.4f}")
    print(f"  HyDE Passage <-> Gold Docs: {mean_cos_hp_g:.4f}")
    print(f"  Delta (HyDE minus Raw)   : {mean_cos_hp_g - mean_cos_q_g:+.4f}")
    if mean_cos_hp_g < mean_cos_q_g:
        print("  -> MECHANISM CONFIRMED: HyDE hypothetical passage is FARTHER from the gold documents than the raw query is!")

if __name__ == "__main__":
    main()
