"""Fix R-06a Sanity Retrievals and Harness Guard (Task R-06a-fix).

Does NOT re-index Qdrant collections.
Updates sanity retrievals in reports/r06a_index.json using valid test queries.
"""

import sys
import json
import logging
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from adaptive_retriever.config import Config
from adaptive_retriever.vectorstore import VectorStore
from adaptive_retriever.benchmarks.beir import (
    is_relevant_beir,
    verify_beir_benchmark_harness,
)

sys.path.insert(0, ".")
from scripts.index_beir_corpora import load_jsonl, load_qrels_tsv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_sanity_retrievals_and_guard(dataset_name: str, queries: list, qrels: dict, cfg_path: str):
    logger.info(f"=== Running Sanity Retrievals & Harness Guard for {dataset_name} ===")
    cfg = Config.load(cfg_path)
    vs = VectorStore.from_config(cfg, collection_base_name=f"beir_{dataset_name}")

    # 1. Harness Startup Guard
    verify_beir_benchmark_harness(queries, qrels, vs, cfg, sample_size=50)

    # 2. 5 Sanity Retrievals
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

    return sanity_retrievals


def main():
    logger.info("=== TASK R-06a-fix DIAGNOSTIC & REPORT UPDATE ===")

    # 1. Load NFCorpus Test Queries (323)
    nf_dir = Path("data/beir/nfcorpus")
    nf_queries_all = load_jsonl(nf_dir / "queries.jsonl")
    nf_qrels = load_qrels_tsv(nf_dir / "qrels" / "test.tsv")
    nf_queries = [q for q in nf_queries_all if str(q["_id"]) in nf_qrels]
    logger.info(f"Loaded {len(nf_queries)} NFCorpus test queries.")

    # 2. Load HotpotQA Test Queries (500)
    p_q = hf_hub_download(repo_id="BeIR/hotpotqa", filename="queries/queries-00000-of-00001.parquet", repo_type="dataset")
    p_qrels = hf_hub_download(repo_id="BeIR/hotpotqa-qrels", filename="test.tsv", repo_type="dataset")
    hp_qrels = load_qrels_tsv(Path(p_qrels))
    q_table = pq.read_table(p_q)
    q_dicts = q_table.to_pydict()
    hp_test_queries = []
    for i in range(len(q_dicts["_id"])):
        qid = str(q_dicts["_id"][i])
        if qid in hp_qrels:
            hp_test_queries.append({"_id": qid, "text": str(q_dicts["text"][i])})
    hp_queries_500 = hp_test_queries[:500]
    logger.info(f"Loaded {len(hp_queries_500)} HotpotQA test queries.")

    # 3. Sanity Retrievals & Guard
    nf_sanity = run_sanity_retrievals_and_guard("nfcorpus", nf_queries, nf_qrels, "config/beir_nfcorpus.yaml")
    hp_sanity = run_sanity_retrievals_and_guard("hotpotqa", hp_queries_500, hp_qrels, "config/beir_hotpotqa.yaml")

    # 4. Check Corpus A Integrity
    cfg_a = Config.load("config/default.yaml")
    vs_a = VectorStore.from_config(cfg_a)
    col_info_a = vs_a.client.get_collection(vs_a.collection_name)
    points_a = col_info_a.points_count
    assert points_a == 47034, f"CORPUS A INTEGRITY FAILURE: Expected 47034 points but found {points_a}!"
    logger.info(f"CORPUS A INTEGRITY VERIFIED: Collection '{vs_a.collection_name}' has exactly {points_a} points.")

    # 5. Update existing reports/r06a_index.json
    report_path = Path("reports/r06a_index.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    report["nfcorpus"]["sanity_retrievals"] = nf_sanity
    report["hotpotqa"]["sanity_retrievals"] = hp_sanity
    report["corpus_a_integrity"] = {
        "collection_name": vs_a.collection_name,
        "expected_points": 47034,
        "actual_points": points_a,
        "status": "PASSED"
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Updated report saved to {report_path}")
    logger.info("=== R-06a-fix COMPLETE ===")


if __name__ == "__main__":
    main()
