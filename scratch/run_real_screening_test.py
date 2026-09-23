import json
import time
from pathlib import Path
import numpy as np
from qdrant_client.http import models as rest

from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import recall_at_k, ndcg_at_k, mrr, bootstrap_ci, bootstrap_difference_ci
from adaptive_retriever.ingest.representations import evaluate_chunk_density, plan_representations, generate_representations
from adaptive_retriever.llm import LocalTransformersLLMClient
from adaptive_retriever.models import EvalItem, Chunk, RepresentationKind
from adaptive_retriever.vectorstore import VectorStore
from adaptive_retriever.funnel.baseline import run_baseline
from adaptive_retriever.funnel.pipeline import run_funnel, warm_up
from adaptive_retriever.funnel.rerank import Reranker

print("=== V4: REAL SCREENING TEST (n=50 SUBSET) ===", flush=True)

t0_total = time.perf_counter()

cfg = Config.load("config/default.yaml")
eval_path = Path("data/eval/eval_set.jsonl")

with open(eval_path, "r", encoding="utf-8") as f:
    all_items = [EvalItem(**json.loads(line)) for line in f if line.strip()]

valid_items = [item for item in all_items if item.provenance != "synthetic_rejected"]

# 1. Select random 50-query subset (seed 42)
rng = np.random.default_rng(42)
subset_indices = rng.choice(len(valid_items), size=50, replace=False)
sample_items = [valid_items[idx] for idx in sorted(subset_indices)]

print(f"Selected {len(sample_items)} evaluation queries for screening subset.", flush=True)

# 2. Extract union of chunks in top-30 candidate pools
with open("reports/candidate_pools_cache.json", "r", encoding="utf-8") as f:
    cand_pools = json.load(f)

unique_chunks_map = {}
for item in sample_items:
    pool = cand_pools.get(item.query, [])[:30]
    for c in pool:
        cid = c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id")
        ctxt = c.get("text") if isinstance(c, dict) else getattr(c, "text")
        if cid and cid not in unique_chunks_map:
            unique_chunks_map[cid] = Chunk(
                chunk_id=cid,
                doc_id=cid.split("_")[0] if "_" in cid else "doc1",
                ordinal=0,
                text=ctxt,
                char_start=0,
                char_end=len(ctxt),
                chunker_used="recursive_512",
                token_count=len(ctxt.split()),
            )

unique_chunks = list(unique_chunks_map.values())
print(f"Unique chunk count in top-30 candidate pools (n=50 queries): {len(unique_chunks)}", flush=True)

# 3. High-tier filtering and representation generation
high_tier_chunks = [c for c in unique_chunks if evaluate_chunk_density(c, cfg) == "high"]
print(f"High-tier chunks receiving multi-representation treatment: {len(high_tier_chunks)} / {len(unique_chunks)}", flush=True)

# Plan generation requests
_, gen_requests = plan_representations(high_tier_chunks, cfg)
print(f"Generation requests planned: {len(gen_requests)}", flush=True)

t_gen_start = time.perf_counter()
cache_path = Path("data/cache_step2_sample")
llm_client = LocalTransformersLLMClient(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir=cache_path,
    device="cuda",
    dtype="float32",
    batch_size=8,
    max_new_tokens=64,
)
failures_log = []
gen_results = generate_representations(high_tier_chunks, llm_client=llm_client, cfg=cfg, failures=failures_log)

t_gen_elapsed = time.perf_counter() - t_gen_start
print(f"Materialized representations in {t_gen_elapsed:.2f} seconds wall-clock.", flush=True)

# 4. ASSERTIONS ON GENERATION QUALITY
non_raw = [r for r in gen_results if r.kind.value != "raw"]
empty_reps = [r for r in non_raw if not r.text or not r.text.strip()]
if empty_reps:
    raise AssertionError(f"FAIL: {len(empty_reps)} generated representations are empty strings!")

distinct_texts = len(set(r.text for r in non_raw))
total_non_raw = len(non_raw)
uniqueness_ratio = distinct_texts / float(total_non_raw) if total_non_raw > 0 else 1.0

print(f"Generation Quality Check:")
print(f"  Total Non-Raw Representations: {total_non_raw}")
print(f"  Empty Representations Count  : {len(empty_reps)}")
print(f"  Distinct Representation Texts: {distinct_texts} / {total_non_raw} ({uniqueness_ratio:.1%})")

if uniqueness_ratio < 0.95:
    raise AssertionError(
        f"FAIL: Uniqueness ratio {uniqueness_ratio:.1%} is below 95% threshold! "
        f"({distinct_texts} distinct out of {total_non_raw})"
    )

print("ASSERTIONS PASSED: No empty representations, distinct text count within 95% threshold.", flush=True)

# 5. Upsert non-raw screening representations into Qdrant
vs = VectorStore.from_config(cfg, timeout=60.0)

print(f"Upserting {len(non_raw)} multi-representations into Qdrant...", flush=True)
reps_by_chunk = {}
for rep in gen_results:
    reps_by_chunk.setdefault(rep.chunk_id, []).append(rep)

upserted_count = 0
for cid, reps in reps_by_chunk.items():
    chunk = unique_chunks_map[cid]
    reps_to_add = [r for r in reps if r.kind.value != "raw"]
    if reps_to_add:
        vs.upsert_representations(chunk, reps_to_add, metadata={"doc_class": getattr(chunk, "doc_class", "unknown")})
        upserted_count += len(reps_to_add)

print(f"Upserted {upserted_count} non-raw representations into Qdrant collection {vs.collection_name}.", flush=True)

rr_model = cfg.raw.get("models", {}).get("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")
rr_device = cfg.raw.get("models", {}).get("llm", {}).get("device", "auto")
reranker = Reranker(model_name=rr_model, device=rr_device)
warm_up(vs, reranker)

print("Running comparative evaluation over 50 screening queries...", flush=True)

base_r10, base_n10, base_mrr = [], [], []
adap_r10, adap_n10, adap_mrr = [], [], []

t_eval_start = time.perf_counter()

for i, item in enumerate(sample_items, start=1):
    # Baseline Arm (Raw-Only)
    cands_base, _ = run_baseline(item.query, vs=vs, cfg=cfg, limit=10)
    b_texts = [c.text for c in cands_base]
    base_r10.append(recall_at_k(b_texts, item.relevant_passages, k=10))
    base_n10.append(ndcg_at_k(b_texts, item.relevant_passages, k=10))
    base_mrr.append(mrr(b_texts, item.relevant_passages))
    
    # Adaptive Arm (Multi-Representation Pipeline)
    cands_ad, _ = run_funnel(item.query, vs=vs, cfg=cfg, reranker=reranker)
    a_texts = [c.text for c in cands_ad]
    adap_r10.append(recall_at_k(a_texts, item.relevant_passages, k=10))
    adap_n10.append(ndcg_at_k(a_texts, item.relevant_passages, k=10))
    adap_mrr.append(mrr(a_texts, item.relevant_passages))
    
    if i % 10 == 0 or i == len(sample_items):
        elapsed = time.perf_counter() - t_eval_start
        print(f"  Processed {i}/{len(sample_items)} queries ({elapsed:.1f}s elapsed)...", flush=True)

# Cleanup screening non-raw points from Qdrant to preserve raw-only index
try:
    vs.client.delete(
        collection_name=vs.collection_name,
        points_selector=rest.Filter(
            must_not=[rest.FieldCondition(key="kind", match=rest.MatchValue(value="raw"))]
        )
    )
    print("Cleaned up non-raw screening points from Qdrant collection.", flush=True)
except Exception as err:
    print(f"Warning: Cleanup of non-raw points failed: {err}", flush=True)

# 6. Compute CIs
n_resamples = 10000
ci_r10_base = bootstrap_ci(base_r10, n=n_resamples, seed=42)
ci_n10_base = bootstrap_ci(base_n10, n=n_resamples, seed=42)
ci_mrr_base = bootstrap_ci(base_mrr, n=n_resamples, seed=42)

ci_r10_adap = bootstrap_ci(adap_r10, n=n_resamples, seed=42)
ci_n10_adap = bootstrap_ci(adap_n10, n=n_resamples, seed=42)
ci_mrr_adap = bootstrap_ci(adap_mrr, n=n_resamples, seed=42)

diff_r10 = bootstrap_difference_ci(adap_r10, base_r10, n=n_resamples, seed=42)
diff_n10 = bootstrap_difference_ci(adap_n10, base_n10, n=n_resamples, seed=42)
diff_mrr = bootstrap_difference_ci(adap_mrr, base_mrr, n=n_resamples, seed=42)

summary = {
    "unique_chunks_count": len(unique_chunks),
    "high_tier_chunks_count": len(high_tier_chunks),
    "generations_count": len(gen_requests),
    "generation_wall_clock_seconds": round(t_gen_elapsed, 2),
    "non_raw_representations_count": total_non_raw,
    "distinct_representation_texts": distinct_texts,
    "uniqueness_ratio": round(uniqueness_ratio, 4),
    "raw_only_baseline": {
        "recall_10": ci_r10_base,
        "ndcg_10": ci_n10_base,
        "mrr": ci_mrr_base,
    },
    "multi_rep_adaptive": {
        "recall_10": ci_r10_adap,
        "ndcg_10": ci_n10_adap,
        "mrr": ci_mrr_adap,
    },
    "difference_adaptive_minus_baseline": {
        "recall_10_diff": diff_r10,
        "ndcg_10_diff": diff_n10,
        "mrr_diff": diff_mrr,
    }
}

with open("scratch/real_screening_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved real screening test results to scratch/real_screening_results.json", flush=True)
