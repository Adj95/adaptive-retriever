"""BEIR Corpora Downloading, Subsampling, Chunking, and Indexing Script (TASK R-06a).

Handles NFCorpus and HotpotQA (subsampled to 20,000 docs with seed 42).
Ingests raw chunks into Qdrant collections beir_nfcorpus_<hash> and beir_hotpotqa_<hash>.
"""

import os
import sys
import json
import zipfile
import urllib.request
import ssl
import random
import time
import logging
from pathlib import Path
import numpy as np
import requests
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from adaptive_retriever.config import Config, compute_hash
from adaptive_retriever.ingest.chunkers import chunk_recursive_512
from adaptive_retriever.ingest.parsers import ParsedDocument
from adaptive_retriever.models import Chunk
from adaptive_retriever.vectorstore import VectorStore, stable_point_id
from adaptive_retriever.benchmarks.beir import is_relevant_beir, get_qrel_score_beir, verify_beir_benchmark_harness
from qdrant_client.http import models as rest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BEIR_DOWNLOAD_URLS = {
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
}

DATA_DIR = Path("data/beir")
DOWNLOADS_DIR = DATA_DIR / "downloads"

def download_and_extract(dataset_name: str) -> Path:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOADS_DIR / f"{dataset_name}.zip"
    extract_dir = DATA_DIR / dataset_name
    
    if extract_dir.exists() and (extract_dir / "corpus.jsonl").exists():
        logger.info(f"Dataset '{dataset_name}' already extracted at {extract_dir}")
        return extract_dir

    url = BEIR_DOWNLOAD_URLS[dataset_name]
    logger.info(f"Downloading {dataset_name} from {url}...")
    resp = requests.get(url, stream=True, verify=False, timeout=60)
    resp.raise_for_status()
    total_size = int(resp.headers.get('content-length', 0))
    downloaded = 0
    with open(zip_path, 'wb') as out_file:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            if chunk:
                out_file.write(chunk)
                downloaded += len(chunk)
    logger.info(f"Downloaded {downloaded} bytes for {dataset_name}.")
    
    logger.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    logger.info(f"Extracted to {extract_dir}")
    return extract_dir

def load_jsonl(file_path: Path):
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def load_qrels_tsv(file_path: Path):
    qrels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        header = next(f, None) # header line: query-id \t corpus-id \t score
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

def prepare_hotpotqa_hf() -> tuple[list[dict], list[dict], dict]:
    logger.info("Preparing HotpotQA dataset via HuggingFace hub parquet streaming...")
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
    num_needed = 20000 - 996  # target 20,000 docs total (996 gold + distractors)

    random.seed(42)
    logger.info(f"Streaming {pf.metadata.num_rows} corpus rows from parquet with reservoir sampling...")
    for batch in pf.iter_batches(batch_size=50000):
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
                if len(distractors_reservoir) < (20000 - len(gold_doc_ids)):
                    distractors_reservoir.append(rec)
                else:
                    r = random.randint(1, distractor_count)
                    if r <= (20000 - len(gold_doc_ids)):
                        distractors_reservoir[r - 1] = rec

    num_needed = 20000 - len(gold_records)
    logger.info(f"Found {len(gold_records)} gold doc records out of 5,233,329 total rows (distractor pool count: {distractor_count})")
    
    subsampled_records = gold_records + distractors_reservoir[:num_needed]
    logger.info(f"Subsampled HotpotQA total docs: {len(subsampled_records)}")
    
    return subsampled_records, queries_500, qrels

def index_corpus(dataset_name: str, corpus_records: list[dict], queries: list[dict], qrels: dict, cfg_path: str):
    logger.info(f"--- Indexing {dataset_name} ---")
    cfg = Config.load(cfg_path)
    
    # 1. Chunk all documents
    t0_chunk = time.perf_counter()
    all_chunks = []
    for idx, rec in enumerate(corpus_records, 1):
        doc_id = str(rec["_id"])
        title = rec.get("title", "")
        text = rec.get("text", "")
        chunks = chunk_beir_doc(doc_id, title, text)
        all_chunks.extend(chunks)
        if idx % 2000 == 0 or idx == len(corpus_records):
            logger.info(f"  Chunked {idx}/{len(corpus_records)} docs ({len(all_chunks)} chunks so far)")
    t1_chunk = time.perf_counter()
    
    # 2. Verify token counts against 480 token cap
    logger.info(f"Verifying {len(all_chunks)} chunks against 480 BGE token cap...")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    char_lens = [len(c.text) for c in all_chunks]
    token_counts = []
    for c in all_chunks:
        t_cnt = len(tok.encode(c.text, add_special_tokens=False))
        c.token_count = t_cnt
        token_counts.append(t_cnt)
        assert t_cnt <= 480, f"HARD ASSERTION FAILED: Chunk {c.chunk_id} has {t_cnt} tokens > 480!"
    
    logger.info(f"SUCCESS: All {len(all_chunks)} chunks pass <= 480 token cap assertion.")
    
    tok_min, tok_p50, tok_p95, tok_p99, tok_max = (
        int(np.min(token_counts)),
        float(np.percentile(token_counts, 50)),
        float(np.percentile(token_counts, 95)),
        float(np.percentile(token_counts, 99)),
        int(np.max(token_counts))
    )
    char_min, char_p50, char_p95, char_p99, char_max = (
        int(np.min(char_lens)),
        float(np.percentile(char_lens, 50)),
        float(np.percentile(char_lens, 95)),
        float(np.percentile(char_lens, 99)),
        int(np.max(char_lens))
    )
    
    logger.info(f"Token distribution (BGE-small): min={tok_min}, p50={tok_p50:.1f}, p95={tok_p95:.1f}, max={tok_max}")
    
    # 3. VectorStore & Collection setup
    vs = VectorStore.from_config(cfg, collection_base_name=cfg.raw["vectorstore"]["collection"])
    logger.info(f"Target Qdrant collection: '{vs.collection_name}'")
    try:
        vs.client.delete_collection(vs.collection_name)
    except Exception:
        pass
    vs._ensure_collection()
    
    # 4. Ingest raw chunks into Qdrant
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
        vs.client.upsert(collection_name=vs.collection_name, points=points)
        total_points += len(points)
        if (i // batch_size + 1) % 50 == 0 or (i + batch_size) >= len(all_chunks):
            logger.info(f"  Indexed {total_points}/{len(all_chunks)} points...")
            
    t1_embed = time.perf_counter()
    embed_time_s = t1_embed - t0_embed
    logger.info(f"Indexed {total_points} points in {embed_time_s:.2f} s ({total_points/embed_time_s:.1f} pts/sec).")
    
    # 4.5 Run Harness Startup Guard (Task R-06a-fix F5)
    logger.info(f"Running BEIR harness startup guard for {dataset_name}...")
    verify_beir_benchmark_harness(queries, qrels, vs, cfg)

    # 5. Sanity Retrievals (5 queries, dense-only, top-3)
    sanity_retrievals = []
    sample_queries = queries[:5]
    logger.info(f"\nRunning 5 dense-only sanity retrievals for {dataset_name}:")
    for q in sample_queries:
        qid = str(q["_id"])
        qtext = q["text"]
        
        q_vec = vs.embedding_client.encode(qtext)
        
        search_hits = vs.search_dense(q_vec, limit=3)
        retrieved_items = []
        for rep_id, payload, score in search_hits:
            chunk_id = payload.get("chunk_id", "")
            doc_id = payload.get("doc_id", "")
            is_rel = is_relevant_beir(payload, qid, qrels)
            retrieved_items.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "score": float(score),
                "is_relevant": is_rel
            })
            
        sanity_item = {
            "query_id": qid,
            "query_text": qtext,
            "top3_retrieved": retrieved_items
        }
        sanity_retrievals.append(sanity_item)
        
        print(f"\nQuery ID [{qid}]: {qtext[:80]}...")
        for rank, item in enumerate(retrieved_items, 1):
            print(f"  Rank {rank}: chunk_id={item['chunk_id']}, doc_id={item['doc_id']}, score={item['score']:.4f}, is_relevant={item['is_relevant']}")

    return {
        "ingest_config_hash": cfg.ingest_config_hash,
        "collection_name": vs.collection_name,
        "total_documents": len(corpus_records),
        "total_chunks": len(all_chunks),
        "total_points": total_points,
        "max_tokens_asserted": tok_max,
        "token_distribution": {
            "min": tok_min,
            "p50": tok_p50,
            "p95": tok_p95,
            "p99": tok_p99,
            "max": tok_max,
        },
        "char_distribution": {
            "min": char_min,
            "p50": char_p50,
            "p95": char_p95,
            "p99": char_p99,
            "max": char_max,
        },
        "indexing_wall_clock_seconds": round(t1_embed - t0_chunk, 2),
        "sanity_retrievals": sanity_retrievals
    }

def main():
    logger.info("=== TASK R-06a BEIR INDEXING ===")
    
    # 1. Prepare NFCorpus
    nf_dir = download_and_extract("nfcorpus")
    nf_corpus_records = load_jsonl(nf_dir / "corpus.jsonl")
    nf_queries_all = load_jsonl(nf_dir / "queries.jsonl")
    nf_qrels = load_qrels_tsv(nf_dir / "qrels" / "test.tsv")
    nf_queries = [q for q in nf_queries_all if str(q["_id"]) in nf_qrels]
    logger.info(f"NFCorpus documents count: {len(nf_corpus_records)}, test queries count: {len(nf_queries)}")
    
    # 2. Prepare HotpotQA via Hugging Face Hub Parquet Streaming
    hp_subsampled_records, hp_queries_500, hp_qrels = prepare_hotpotqa_hf()

    # 3. Index NFCorpus
    nf_stats = index_corpus("nfcorpus", nf_corpus_records, nf_queries, nf_qrels, "config/beir_nfcorpus.yaml")
    
    # 4. Index HotpotQA
    hp_stats = index_corpus("hotpotqa", hp_subsampled_records, hp_queries_500, hp_qrels, "config/beir_hotpotqa.yaml")

    # 5. Check Corpus A integrity
    cfg_a = Config.load("config/default.yaml")
    vs_a = VectorStore.from_config(cfg_a)
    col_info_a = vs_a.client.get_collection(vs_a.collection_name)
    points_a = col_info_a.points_count
    assert points_a == 47034, f"CORPUS A INTEGRITY FAILURE: Expected 47034 points but found {points_a}!"
    logger.info(f"CORPUS A INTEGRITY VERIFIED: Collection '{vs_a.collection_name}' has exactly {points_a} points.")
    
    corpus_a_integrity = {
        "collection_name": vs_a.collection_name,
        "expected_points": 47034,
        "actual_points": points_a,
        "status": "PASSED"
    }

    # 6. Generate final report
    report = {
        "task": "R-06a",
        "nfcorpus": nf_stats,
        "hotpotqa": {
            **hp_stats,
            "subsampling_rule": "500 test queries gold docs + random distractors (seed 42) = 20,000 docs total"
        },
        "corpus_a_integrity": corpus_a_integrity
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "r06a_index.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Report saved to {report_path}")
    logger.info("=== INDEXING AND VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main()
