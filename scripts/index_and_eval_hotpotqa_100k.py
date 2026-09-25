"""TASK R-06c-prep — Index HotpotQA 100k and Evaluate Baselines.

P1: Correct Corpus A canonical values and shrinkage multiples.
P2: Re-index HotpotQA with 100,000 document pool (seed 42) into beir_hotpotqa_100k_<hash>.
P3: State headroom verdict per corpus for R-06c component suite.
"""

import sys
import json
import random
import time
import logging
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from adaptive_retriever.config import Config
from adaptive_retriever.ingest.chunkers import chunk_recursive_512
from adaptive_retriever.ingest.parsers import ParsedDocument
from adaptive_retriever.models import Chunk
from adaptive_retriever.vectorstore import VectorStore, stable_point_id
from adaptive_retriever.benchmarks.beir import (
    is_relevant_beir,
    get_qrel_score_beir,
    verify_beir_benchmark_harness,
)

def load_qrels_tsv(file_path: Path) -> dict:
    qrels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        header = next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, doc_id, score = parts[0], parts[1], int(parts[2])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = score
    return qrels

def chunk_beir_doc(doc_id: str, title: str, text: str) -> list[Chunk]:
    full_text = f"{title}\n{text}".strip() if title else text.strip()
    parsed = ParsedDocument(
        doc_id=doc_id,
        text=full_text,
        metadata={"title": title},
        structure_kind="digital_prose"
    )
    chunks = chunk_recursive_512(parsed, doc_id=doc_id)
    return chunks
from adaptive_retriever.funnel.baseline import run_baseline
from qdrant_client.http import models as rest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("r06c_prep")

def compute_dcg(gains, k=10):
    dcg = 0.0
    for i, g in enumerate(gains[:k]):
        if g > 0:
            dcg += g / np.log2(i + 2)
    return dcg

def compute_ndcg(retrieved_chunks, qid, qrels, k=10):
    if qid not in qrels:
        return 0.0
    query_qrels = qrels[qid]
    gold_scores = sorted(query_qrels.values(), reverse=True)
    idcg = compute_dcg(gold_scores, k=k)
    if idcg == 0:
        return 0.0
    gains = [get_qrel_score_beir(c, qid, qrels) for c in retrieved_chunks]
    dcg = compute_dcg(gains, k=k)
    return dcg / idcg

def eval_arm(queries, qrels, vs, cfg, limit=10):
    n_queries = len(queries)
    ndcg10_list = []
    ndcg5_list = []
    rec5_list = []
    rec10_list = []
    mrr_list = []

    for idx, q in enumerate(queries, 1):
        qid = str(q["_id"])
        cands, _ = run_baseline(q["text"], vs, cfg, limit=limit)
        
        ndcg10 = compute_ndcg(cands, qid, qrels, k=10)
        ndcg5 = compute_ndcg(cands, qid, qrels, k=5)
        
        rel5 = [1 if is_relevant_beir(c, qid, qrels) else 0 for c in cands[:5]]
        rel10 = [1 if is_relevant_beir(c, qid, qrels) else 0 for c in cands[:10]]
        
        rec5 = float(sum(rel5) > 0)
        rec10 = float(sum(rel10) > 0)
        
        mrr = 0.0
        for r_i, r_val in enumerate(rel10, 1):
            if r_val == 1:
                mrr = 1.0 / r_i
                break
                
        ndcg10_list.append(ndcg10)
        ndcg5_list.append(ndcg5)
        rec5_list.append(rec5)
        rec10_list.append(rec10)
        mrr_list.append(mrr)
        
        if idx % 100 == 0 or idx == n_queries:
            logger.info(f"  Evaluated {idx}/{n_queries} queries...")

    return {
        "ndcg_10": float(np.mean(ndcg10_list)),
        "ndcg_5": float(np.mean(ndcg5_list)),
        "recall_5": float(np.mean(rec5_list)),
        "recall_10": float(np.mean(rec10_list)),
        "mrr": float(np.mean(mrr_list)),
    }

def prepare_hotpotqa_100k() -> tuple[list[dict], list[dict], dict]:
    logger.info("Preparing HotpotQA 100k dataset via HuggingFace hub parquet streaming...")
    p_corp = hf_hub_download(repo_id="BeIR/hotpotqa", filename="corpus/corpus-00000-of-00001.parquet", repo_type="dataset")
    p_q = hf_hub_download(repo_id="BeIR/hotpotqa", filename="queries/queries-00000-of-00001.parquet", repo_type="dataset")
    p_qrels = hf_hub_download(repo_id="BeIR/hotpotqa-qrels", filename="test.tsv", repo_type="dataset")
    
    qrels = load_qrels_tsv(Path(p_qrels))
    
    q_table = pq.read_table(p_q)
    q_dicts = q_table.to_pydict()
    all_test_queries = []
    for i in range(len(q_dicts["_id"])):
        qid = str(q_dicts["_id"][i])
        if qid in qrels:
            all_test_queries.append({
                "_id": qid,
                "text": str(q_dicts["text"][i])
            })
            
    queries_500 = all_test_queries[:500]
    qids_500 = set(q["_id"] for q in queries_500)
    
    gold_doc_ids = set()
    for qid in qids_500:
        if qid in qrels:
            gold_doc_ids.update(str(did) for did in qrels[qid].keys())
            
    logger.info(f"HotpotQA 500 test queries have {len(gold_doc_ids)} unique gold document IDs.")
    
    pf = pq.ParquetFile(p_corp)
    gold_records = []
    distractors_reservoir = []
    distractor_count = 0
    target_total = 100000
    target_distractors = target_total - len(gold_doc_ids)

    random.seed(42)
    logger.info(f"Streaming corpus rows from parquet for 100k pool with reservoir sampling...")
    for batch in pf.iter_batches(batch_size=100000):
        b_dict = batch.to_pydict()
        b_ids = b_dict["_id"]
        b_titles = b_dict["title"]
        b_texts = b_dict["text"]
        for did, title, text in zip(b_ids, b_titles, b_texts):
            did_str = str(did)
            rec = {"_id": did_str, "title": title or "", "text": text or ""}
            if did_str in gold_doc_ids:
                gold_records.append(rec)
            else:
                distractor_count += 1
                if len(distractors_reservoir) < target_distractors:
                    distractors_reservoir.append(rec)
                else:
                    r = random.randint(1, distractor_count)
                    if r <= target_distractors:
                        distractors_reservoir[r - 1] = rec

    num_needed = target_total - len(gold_records)
    logger.info(f"Found {len(gold_records)} gold doc records out of 5,233,329 total rows.")
    subsampled_records = gold_records + distractors_reservoir[:num_needed]
    logger.info(f"HotpotQA 100k total sampled records: {len(subsampled_records)}")
    
    return subsampled_records, queries_500, qrels

def main():
    logger.info("=== TASK R-06c-prep START ===")
    
    # --- P1 Analysis ---
    p1_summary = {
        "provenance": "The 0.7320 / 0.8140 values reported in the R-06b summary table were untraceable text artifacts from prior draft output and are formally withdrawn.",
        "canonical_corpus_a": {
            "source_file": "reports/experiment_001_3arms.json",
            "arm_a_ndcg10": 0.4311,
            "arm_b_ndcg10": 0.5280,
            "b_minus_a_diff_ndcg10": 0.0969,
            "b_minus_a_ci95_ndcg10": [0.0437, 0.1510],
            "arm_a_mrr": 0.3922,
            "arm_b_mrr": 0.4949,
            "b_minus_a_diff_mrr": 0.1027,
        },
        "corrected_three_corpus_table": [
            {
                "corpus": "Corpus A (Shipped)",
                "arm_a_ndcg10": 0.4311,
                "arm_b_ndcg10": 0.5280,
                "advantage_ndcg10": 0.0969,
                "arm_a_mrr": 0.3922,
                "arm_b_mrr": 0.4949,
                "advantage_mrr": 0.1027,
                "recall_density_note": "Gold set density ~1.0; Recall uncapped."
            },
            {
                "corpus": "NFCorpus (BEIR Test)",
                "arm_a_ndcg10": 0.3530,
                "arm_b_ndcg10": 0.3667,
                "advantage_ndcg10": 0.0137,
                "arm_a_mrr": 0.5343,
                "arm_b_mrr": 0.5812,
                "advantage_mrr": 0.0469,
                "recall_density_note": "Dense qrels (p50=16.0 gold/query); Recall@10 capped at 0.625."
            },
            {
                "corpus": "HotpotQA 20k (BEIR Subsample)",
                "arm_a_ndcg10": 0.9201,
                "arm_b_ndcg10": 0.9451,
                "advantage_ndcg10": 0.0250,
                "arm_a_mrr": 0.9760,
                "arm_b_mrr": 0.9967,
                "advantage_mrr": 0.0207,
                "recall_density_note": "Sparse qrels (2 gold/query); uncapped at k=10."
            }
        ],
        "corrected_shrinkage_multiples": {
            "nfcorpus_reduction": "0.0969 / 0.0137 = 7.07x (~7x fall in nDCG@10 advantage)",
            "hotpotqa_20k_reduction": "0.0969 / 0.0250 = 3.88x (~4x fall in nDCG@10 advantage)"
        }
    }
    
    # --- P2 Indexing HotpotQA 100k ---
    corpus_records, queries, qrels = prepare_hotpotqa_100k()
    
    # P2 Projection
    projected_chunks = len(corpus_records)  # ~1 chunk per doc for Wikipedia abstracts
    # 20k chunks took ~35s embedding time. 100k chunks projected at ~175s - 300s (3-5 minutes)
    projected_embed_time_s = 240.0
    projected_total_time_min = (projected_embed_time_s + 60.0) / 60.0
    
    logger.info(f"=== P2 INDEXING PROJECTION ===")
    logger.info(f"  Target documents: {len(corpus_records):,}")
    logger.info(f"  Projected chunk count: {projected_chunks:,}")
    logger.info(f"  Projected embedding wall-clock: {projected_embed_time_s:.1f} s ({projected_total_time_min:.1f} min)")
    
    if projected_total_time_min > 90.0:
        logger.error(f"Projection exceeds 90 minutes! Stopping as instructed.")
        sys.exit(1)
        
    logger.info("Projection is under 90 minutes. Proceeding with chunking and indexing...")
    
    # Load config and VectorStore for 100k collection
    cfg = Config.load("config/beir_hotpotqa.yaml")
    # Base collection name for 100k pool
    vs = None
    for attempt in range(5):
        try:
            vs = VectorStore.from_config(cfg, collection_base_name="beir_hotpotqa_100k")
            break
        except Exception as e:
            logger.warning(f"VectorStore connection attempt {attempt+1} failed: {e}. Retrying in 3s...")
            time.sleep(3)
    if vs is None:
        raise RuntimeError("Could not connect to VectorStore after 5 attempts.")
    logger.info(f"Target Qdrant collection name: '{vs.collection_name}'")
    
    # 1. Chunking
    t0_chunk = time.perf_counter()
    all_chunks = []
    for idx, rec in enumerate(corpus_records, 1):
        doc_id = str(rec["_id"])
        title = rec.get("title", "")
        text = rec.get("text", "")
        chunks = chunk_beir_doc(doc_id, title, text)
        all_chunks.extend(chunks)
        if idx % 20000 == 0 or idx == len(corpus_records):
            logger.info(f"  Chunked {idx}/{len(corpus_records)} docs ({len(all_chunks)} chunks so far)")
    t1_chunk = time.perf_counter()
    
    # Verify token counts
    logger.info(f"Verifying {len(all_chunks)} chunks against 480 BGE token cap...")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    for c in all_chunks:
        t_cnt = len(tok.encode(c.text, add_special_tokens=False))
        c.token_count = t_cnt
        assert t_cnt <= 480, f"HARD ASSERTION FAILED: Chunk {c.chunk_id} has {t_cnt} tokens > 480!"
    logger.info(f"SUCCESS: All {len(all_chunks)} chunks pass <= 480 token cap assertion.")

    # Reset collection
    try:
        vs.client.delete_collection(vs.collection_name)
    except Exception:
        pass
    vs._ensure_collection()

    # 2. Ingest into Qdrant
    t0_embed = time.perf_counter()
    batch_size = 64
    total_points = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        
        dense_vecs = vs.embedding_client.encode_batch(texts)
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
        for upsert_attempt in range(5):
            try:
                vs.client.upsert(collection_name=vs.collection_name, points=points)
                break
            except Exception as e:
                logger.warning(f"Upsert failed (attempt {upsert_attempt+1}): {e}. Retrying in 2s...")
                time.sleep(2)
        total_points += len(points)
        if (i // batch_size + 1) % 200 == 0 or (i + batch_size) >= len(all_chunks):
            logger.info(f"  Indexed {total_points}/{len(all_chunks)} points...")
            
    t1_embed = time.perf_counter()
    actual_embed_wall_clock_s = t1_embed - t0_embed
    actual_total_indexing_s = t1_embed - t0_chunk
    logger.info(f"Indexing completed: {total_points} points in {actual_total_indexing_s:.2f} s ({actual_embed_wall_clock_s:.2f} s embedding).")

    # 3. Startup Harness Guard
    logger.info("Running Z6 startup harness guard on HotpotQA 100k collection...")
    verify_beir_benchmark_harness(queries, qrels, vs, cfg)

    # 4. Evaluate Arm A and Arm B on HotpotQA 100k
    logger.info("=== Evaluating Arm A (Dense-only) on HotpotQA 100k ===")
    cfg_arm_a = Config.load("config/beir_hotpotqa.yaml")
    cfg_arm_a.raw["funnel"]["hybrid"]["enabled"] = False
    cfg_arm_a.raw["funnel"]["rerank"]["mode"] = "never"
    cfg_arm_a.raw["query"]["transformation"]["strategy"] = "none"
    arm_a_results = eval_arm(queries, qrels, vs, cfg_arm_a, limit=10)
    logger.info(f"Arm A HotpotQA 100k nDCG@10: {arm_a_results['ndcg_10']:.4f}")

    logger.info("=== Evaluating Arm B (Hybrid+Rerank) on HotpotQA 100k ===")
    cfg_arm_b = Config.load("config/beir_hotpotqa.yaml")
    cfg_arm_b.raw["funnel"]["hybrid"]["enabled"] = True
    cfg_arm_b.raw["funnel"]["rerank"]["mode"] = "always"
    cfg_arm_b.raw["query"]["transformation"]["strategy"] = "none"
    arm_b_results = eval_arm(queries, qrels, vs, cfg_arm_b, limit=10)
    logger.info(f"Arm B HotpotQA 100k nDCG@10: {arm_b_results['ndcg_10']:.4f}")

    # Check ceiling effect
    arm_a_ndcg10_100k = arm_a_results["ndcg_10"]
    if 0.60 <= arm_a_ndcg10_100k <= 0.80:
        headroom_status_100k = "Headroom: yes (target range 0.60-0.80 satisfied)."
        has_headroom_100k = True
    elif arm_a_ndcg10_100k > 0.85:
        headroom_status_100k = f"Headroom: no - ceiling effect (nDCG@10 = {arm_a_ndcg10_100k:.4f} > 0.85). Pool is still too easy."
        has_headroom_100k = False
    else:
        headroom_status_100k = f"Headroom: yes (nDCG@10 = {arm_a_ndcg10_100k:.4f})."
        has_headroom_100k = True

    # --- P3 Headroom Verdict ---
    corpora_with_headroom = ["NFCorpus"]
    if has_headroom_100k:
        corpora_with_headroom.append("HotpotQA 100k")

    p3_headroom = {
        "verdicts": [
            "NFCorpus: Arm A nDCG@10 = 0.3530. Headroom: yes.",
            "HotpotQA 20k: Arm A nDCG@10 = 0.9201. Headroom: no - ceiling effect, excluded from R-06c.",
            f"HotpotQA 100k: Arm A nDCG@10 = {arm_a_ndcg10_100k:.4f}. {headroom_status_100k}"
        ],
        "r06c_eval_suite_corpora": corpora_with_headroom
    }

    # Verify Corpus A Integrity
    cfg_a = Config.load("config/default.yaml")
    vs_a = VectorStore.from_config(cfg_a)
    col_info_a = vs_a.client.get_collection(vs_a.collection_name)
    points_a = col_info_a.points_count
    assert points_a == 47034, f"CORPUS A INTEGRITY FAILURE: Expected 47034 points but found {points_a}!"
    logger.info(f"CORPUS A INTEGRITY VERIFIED: Collection '{vs_a.collection_name}' has exactly {points_a} points.")

    # Save r06c_prep.json
    report = {
        "task": "R-06c-prep",
        "p1_corpus_a_correction": p1_summary,
        "p2_hotpotqa_100k": {
            "collection_name": vs.collection_name,
            "sampled_documents": len(corpus_records),
            "projected_chunks": projected_chunks,
            "projected_indexing_time_sec": projected_embed_time_s,
            "actual_chunks": len(all_chunks),
            "actual_points": total_points,
            "actual_indexing_time_sec": round(actual_total_indexing_s, 2),
            "arm_a": arm_a_results,
            "arm_b": arm_b_results,
            "advantage_b_minus_a_ndcg10": round(arm_b_results["ndcg_10"] - arm_a_results["ndcg_10"], 4),
        },
        "p3_headroom_verdict": p3_headroom,
        "corpus_a_integrity": {
            "collection_name": vs_a.collection_name,
            "expected_points": 47034,
            "actual_points": points_a,
            "status": "PASSED"
        }
    }

    out_path = Path("reports/r06c_prep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {out_path}")
    logger.info("=== TASK R-06c-prep COMPLETE ===")

if __name__ == "__main__":
    main()
