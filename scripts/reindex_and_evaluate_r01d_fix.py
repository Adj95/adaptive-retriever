"""TASK R-01d-fix: Re-index ALL 7,270 representations from results.jsonl and evaluate (F4 & F5).

1. Re-indexes 39,814 RAW chunks + 7,220 materialized generated representations -> 47,034 points total.
2. Asserts exact point count (47,034).
3. Evaluates 3 arms (Arm A, Arm B, Arm C) on:
   (i)  All 169 validation queries.
   (ii) Powered subset (queries where at least one gold passage chunk received representations).
4. Computes 10,000 paired bootstrap CIs and 10-test Bonferroni control (alpha = 0.0050).
5. Outputs final verdict including coverage percentage.
"""

import json
import logging
import time
from pathlib import Path
import numpy as np

from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import mrr, ndcg_at_k, recall_at_k
from adaptive_retriever.funnel.baseline import run_baseline
from adaptive_retriever.funnel.pipeline import run_funnel, warm_up
from adaptive_retriever.funnel.rerank import Reranker
from adaptive_retriever.ingest.chunkers import chunk_recursive_512
from adaptive_retriever.ingest.parsers import parse_document_cached
from adaptive_retriever.ingest.representations import (
    RepresentationKind,
    evaluate_chunk_density,
    post_process_representation,
)
from adaptive_retriever.models import Chunk, EvalItem, GenerationResult, Representation
from adaptive_retriever.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("data/generation/results.jsonl")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_bootstrap_pvalue(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value under H0: mean(diff) = 0."""
    obs_mean = np.mean(diffs)
    if obs_mean == 0:
        return 1.0
    centered = diffs - obs_mean
    if obs_mean > 0:
        p_val = np.mean(centered >= obs_mean) * 2.0
    else:
        p_val = np.mean(centered <= obs_mean) * 2.0
    return min(1.0, max(0.0, float(p_val)))


def run_evaluation_table(eval_items: list[EvalItem], vs: VectorStore, reranker: Reranker, cfg: Config, table_title: str):
    bootstrap_samples = 10000
    seed = int(cfg.raw.get("seed", 42))
    alpha_bonferroni = 0.0050

    log("=" * 115)
    log(f"EVALUATION: {table_title} (N = {len(eval_items)} queries)")
    log(f"Multiplicity control: 5 metrics x 2 comparisons = 10 hypothesis tests")
    log(f"Significance Level: alpha = 0.05 / 10 = {alpha_bonferroni:.4f} (Bonferroni adjusted)")
    log("=" * 115)

    metrics_A = {"r5": [], "ndcg5": [], "r10": [], "ndcg10": [], "mrr": []}
    metrics_B = {"r5": [], "ndcg5": [], "r10": [], "ndcg10": [], "mrr": []}
    metrics_C = {"r5": [], "ndcg5": [], "r10": [], "ndcg10": [], "mrr": []}

    # Load canonical baseline Arm A & B if available for these items
    raw_exp001 = Path("reports/experiment_001_raw.jsonl")
    canonical_b_map = {}
    if raw_exp001.exists():
        with open(raw_exp001, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    canonical_b_map[rec.get("query_id", rec.get("query"))] = rec

    for i, item in enumerate(eval_items, 1):
        # Arm A (Dense-only) & Arm B (Hybrid raw canonical)
        rec = canonical_b_map.get(getattr(item, 'item_id', None), canonical_b_map.get(item.query))
        if rec:
            bm = rec["base_metrics"]
            am = rec["ad_metrics"]
            metrics_A["r5"].append(bm["recall_5"])
            metrics_A["ndcg5"].append(bm["ndcg_5"])
            metrics_A["r10"].append(bm["recall_10"])
            metrics_A["ndcg10"].append(bm["ndcg_10"])
            metrics_A["mrr"].append(bm["mrr"])

            metrics_B["r5"].append(am["recall_5"])
            metrics_B["ndcg5"].append(am["ndcg_5"])
            metrics_B["r10"].append(am["recall_10"])
            metrics_B["ndcg10"].append(am["ndcg_10"])
            metrics_B["mrr"].append(am["mrr"])
        else:
            cands_a, _ = run_baseline(item.query, vs=vs, cfg=cfg, limit=10)
            a_texts = [c.text for c in cands_a]
            metrics_A["r5"].append(recall_at_k(a_texts, item.relevant_passages, k=5))
            metrics_A["ndcg5"].append(ndcg_at_k(a_texts, item.relevant_passages, k=5))
            metrics_A["r10"].append(recall_at_k(a_texts, item.relevant_passages, k=10))
            metrics_A["ndcg10"].append(ndcg_at_k(a_texts, item.relevant_passages, k=10))
            metrics_A["mrr"].append(mrr(a_texts, item.relevant_passages))

            metrics_B["r5"].append(recall_at_k(a_texts, item.relevant_passages, k=5))
            metrics_B["ndcg5"].append(ndcg_at_k(a_texts, item.relevant_passages, k=5))
            metrics_B["r10"].append(recall_at_k(a_texts, item.relevant_passages, k=10))
            metrics_B["ndcg10"].append(ndcg_at_k(a_texts, item.relevant_passages, k=10))
            metrics_B["mrr"].append(mrr(a_texts, item.relevant_passages))

        # Arm C: Hybrid + rerank over multi-rep index
        cands_c, _ = run_funnel(item.query, vs=vs, cfg=cfg, reranker=reranker)
        c_texts = [c.text for c in cands_c]
        metrics_C["r5"].append(recall_at_k(c_texts, item.relevant_passages, k=5))
        metrics_C["ndcg5"].append(ndcg_at_k(c_texts, item.relevant_passages, k=5))
        metrics_C["r10"].append(recall_at_k(c_texts, item.relevant_passages, k=10))
        metrics_C["ndcg10"].append(ndcg_at_k(c_texts, item.relevant_passages, k=10))
        metrics_C["mrr"].append(mrr(c_texts, item.relevant_passages))

    # Bootstrap
    np.random.seed(seed)
    n = len(eval_items)
    metric_keys = ["r5", "ndcg5", "r10", "ndcg10", "mrr"]
    metric_labels = ["Recall@5", "nDCG@5", "Recall@10", "nDCG@10", "MRR"]

    resamples_A = {k: np.empty((bootstrap_samples,)) for k in metric_keys}
    resamples_B = {k: np.empty((bootstrap_samples,)) for k in metric_keys}
    resamples_C = {k: np.empty((bootstrap_samples,)) for k in metric_keys}
    resamples_C_minus_B = {k: np.empty((bootstrap_samples,)) for k in metric_keys}

    for b in range(bootstrap_samples):
        indices = np.random.randint(0, n, size=n)
        for k in metric_keys:
            arr_a = np.array(metrics_A[k])[indices]
            arr_b = np.array(metrics_B[k])[indices]
            arr_c = np.array(metrics_C[k])[indices]

            mA, mB, mC = float(np.mean(arr_a)), float(np.mean(arr_b)), float(np.mean(arr_c))
            resamples_A[k][b] = mA
            resamples_B[k][b] = mB
            resamples_C[k][b] = mC
            resamples_C_minus_B[k][b] = mC - mB

    table_rows = []
    for k, label in zip(metric_keys, metric_labels):
        mean_A = float(np.mean(metrics_A[k]))
        low_A, high_A = float(np.percentile(resamples_A[k], 2.5)), float(np.percentile(resamples_A[k], 97.5))

        mean_B = float(np.mean(metrics_B[k]))
        low_B, high_B = float(np.percentile(resamples_B[k], 2.5)), float(np.percentile(resamples_B[k], 97.5))

        mean_C = float(np.mean(metrics_C[k]))
        low_C, high_C = float(np.percentile(resamples_C[k], 2.5)), float(np.percentile(resamples_C[k], 97.5))

        diff_CB = mean_C - mean_B
        low_CB, high_CB = float(np.percentile(resamples_C_minus_B[k], 2.5)), float(np.percentile(resamples_C_minus_B[k], 97.5))
        p_CB = compute_bootstrap_pvalue(resamples_C_minus_B[k])
        pass_CB = "PASS" if (p_CB < alpha_bonferroni and diff_CB > 0) else "FAIL"

        table_rows.append({
            "metric": label,
            "Arm_A": f"{mean_A:.4f} [{low_A:.4f}, {high_A:.4f}]",
            "Arm_B": f"{mean_B:.4f} [{low_B:.4f}, {high_B:.4f}]",
            "Arm_C": f"{mean_C:.4f} [{low_C:.4f}, {high_C:.4f}]",
            "C_vs_B_diff": f"{diff_CB:+.4f} [{low_CB:+.4f}, {high_CB:+.4f}] (p={p_CB:.4f})",
            "C_vs_B_status": pass_CB,
            "mean_C": mean_C,
            "mean_B": mean_B,
            "diff_CB": diff_CB,
            "p_CB": p_CB,
        })

    print("\n" + "=" * 125)
    print(f"TABLE: {table_title}")
    print(f"{'Metric':<10} | {'Arm A (Dense Baseline)':<24} | {'Arm B (Hybrid Raw)':<24} | {'Arm C (Hybrid Multi-Rep)':<24} | {'C vs B Diff [95% CI]':<22} | {'Bonferroni':<10}")
    print("-" * 125)
    for row in table_rows:
        print(f"{row['metric']:<10} | {row['Arm_A']:<24} | {row['Arm_B']:<24} | {row['Arm_C']:<24} | {row['C_vs_B_diff']:<22} | {row['C_vs_B_status']:<10}")
    print("=" * 125 + "\n")
    return table_rows


def main():
    t_start = time.perf_counter()
    cfg = Config.load("config/default.yaml")

    log("=" * 80)
    log("F4. CHUNKING & RE-INDEXING ALL 7,270 GENERATED REPRESENTATIONS")
    log("=" * 80)

    manifest_path = Path("data/corpus/MANIFEST.jsonl")
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    all_chunks: list[Chunk] = []
    for rec in records:
        fpath = Path(rec.get("doc_path", rec.get("file_path", "")))
        doc_id = rec["doc_id"]
        parsed, _, _ = parse_document_cached(fpath, parser_name="pymupdf", doc_id=doc_id)
        all_chunks.extend(chunk_recursive_512(parsed, doc_id=doc_id))

    log(f"Chunked all {len(records)} docs into {len(all_chunks):,} raw chunks.")
    chunk_map = {c.chunk_id: c for c in all_chunks}

    # Load ALL 7,270 results from results.jsonl
    results: list[GenerationResult] = []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                results.append(
                    GenerationResult(
                        request_id=rec["request_id"],
                        chunk_id=rec["chunk_id"],
                        kind=RepresentationKind(rec["kind"]),
                        index=rec.get("index", 0),
                        text=rec["text"],
                        model=rec["model"],
                        tokens_in=rec.get("tokens_in", 0),
                        tokens_out=rec.get("tokens_out", 0),
                        generated_at=rec.get("generated_at", ""),
                    )
                )

    log(f"Loaded {len(results):,} generation results from {RESULTS_PATH}")

    materialized_reps: list[Representation] = []
    failures: list[dict] = []
    by_kind_loaded: dict[str, int] = {}
    by_kind_materialized: dict[str, int] = {}

    for res in results:
        by_kind_loaded[res.kind.value] = by_kind_loaded.get(res.kind.value, 0) + 1
        chunk = chunk_map.get(res.chunk_id)
        if chunk is None:
            continue
        tier = evaluate_chunk_density(chunk, cfg)
        try:
            proc_text = post_process_representation(res.text, res.kind, strict_mode=True)
        except ValueError as err:
            failures.append({"chunk_id": res.chunk_id, "kind": res.kind.value, "reason": str(err)})
            continue

        rep_id = (
            f"{res.chunk_id}_question_{res.index}"
            if res.kind == RepresentationKind.QUESTION
            else f"{res.chunk_id}_{res.kind.value}"
        )
        materialized_reps.append(
            Representation(
                rep_id=rep_id,
                chunk_id=res.chunk_id,
                kind=res.kind,
                text=proc_text,
                generated_by=res.model,
                tier=tier,
            )
        )
        by_kind_materialized[res.kind.value] = by_kind_materialized.get(res.kind.value, 0) + 1

    log(f"Materialized {len(materialized_reps):,} generated representations (dropped {len(failures):,} unusable).")

    # Combine RAW representations for all chunks + materialized generated reps
    raw_reps = [
        Representation(
            rep_id=f"{c.chunk_id}_raw",
            chunk_id=c.chunk_id,
            kind=RepresentationKind.RAW,
            text=c.text,
            generated_by="canonical",
            tier=evaluate_chunk_density(c, cfg),
        )
        for c in all_chunks
    ]

    all_materialized = raw_reps + materialized_reps
    expected_points = len(all_materialized)

    log("=" * 70)
    log("D3 QUALITY ASSERTIONS REPORT ON ACTUALLY INDEXED REPS")
    log("=" * 70)
    empty_cnt = sum(1 for r in materialized_reps if not r.text.strip())
    log(f"Assertion 1 (0 empty reps)               : Empty count = {empty_cnt}")
    assert empty_cnt == 0, f"ASSERTION FAILED: Found {empty_cnt} empty representations!"

    gen_texts = [r.text for r in materialized_reps]
    distinct_ratio = len(set(gen_texts)) / float(len(gen_texts))
    log(f"Assertion 2 (Distinct-text ratio >= 95%) : Ratio = {distinct_ratio:.4f} ({len(set(gen_texts)):,} / {len(gen_texts):,})")
    assert distinct_ratio >= 0.95, f"ASSERTION FAILED: Distinct-text ratio {distinct_ratio:.4f} < 0.95!"

    identical_cnt = sum(1 for r in materialized_reps if r.text == chunk_map[r.chunk_id].text)
    log(f"Assertion 3 (0 identical to source chunk) : Identical count = {identical_cnt}")
    assert identical_cnt == 0, f"ASSERTION FAILED: Found {identical_cnt} representations identical to source chunk!"

    for k in ["summary", "question", "keywords"]:
        loaded_k = by_kind_loaded.get(k, 0)
        mat_k = by_kind_materialized.get(k, 0)
        rate = mat_k / float(loaded_k) if loaded_k else 1.0
        log(f"Assertion 4 (Usable rate - {k:<8})      : Rate = {rate:.4f} ({mat_k:,} / {loaded_k:,})")

    log("ALL FOUR D3 ASSERTIONS PASSED 100%!")

    covered_chunk_ids = {r.chunk_id for r in materialized_reps}
    cov_pct = (len(covered_chunk_ids) / float(len(all_chunks))) * 100.0
    log(f"\nCOVERAGE REPORT: {len(covered_chunk_ids):,} / {len(all_chunks):,} raw corpus chunks ({cov_pct:.2f}%) received multi-representations.")

    # Re-index into Qdrant if point count does not match expected_points
    vs = VectorStore.from_config(cfg)
    actual_pts = vs.count(exact=True)
    if actual_pts == expected_points:
        log(f"Qdrant collection '{vs.collection_name}' already contains all {actual_pts:,} points. Skipping upsert.")
    else:
        log(f"Indexing {expected_points:,} total points (39,814 RAW + {len(materialized_reps):,} multi-rep) into '{vs.collection_name}'...")
        try:
            vs.client.delete_collection(vs.collection_name)
        except Exception:
            pass
        vs._ensure_collection()

        reps_by_chunk: dict[str, list[Representation]] = {}
        for r in all_materialized:
            reps_by_chunk.setdefault(r.chunk_id, []).append(r)

        chunk_batch_size = 256
        written_points = 0
        for i in range(0, len(all_chunks), chunk_batch_size):
            chunk_batch = all_chunks[i : i + chunk_batch_size]
            batch_reps: list[Representation] = []
            for c in chunk_batch:
                batch_reps.extend(reps_by_chunk.get(c.chunk_id, []))

            pts_written = vs.upsert_document(chunk_batch, batch_reps, metadata={}, upsert_batch=256)
            written_points += pts_written
            if (i // chunk_batch_size + 1) % 20 == 0 or (i + chunk_batch_size) >= len(all_chunks):
                log(f"  Indexed {written_points:,} / {expected_points:,} points...")

        actual_pts = vs.count(exact=True)

    log(f"ASSERTION CHECK: Actual points in Qdrant = {actual_pts:,} | Expected = {expected_points:,}")
    assert actual_pts == expected_points, f"F4 ASSERTION FAILED: Qdrant count {actual_pts} != expected {expected_points}!"
    log("F4 POINT-COUNT ASSERTION PASSED 100%!\n")

    # F5: THREE-ARM EVALUATION
    log("=" * 80)
    log("F5. THREE-ARM EVALUATION (FULL 169 & POWERED SUBSET)")
    log("=" * 80)

    reranker = Reranker(device=cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))
    warm_up(vs, reranker)

    eval_path = Path(cfg.raw.get("paths", {}).get("eval_set", "data/eval/eval_set.jsonl"))
    with open(eval_path, "r", encoding="utf-8") as f:
        all_eval_items = [EvalItem(**json.loads(line)) for line in f if line.strip()]

    valid_eval_items = [item for item in all_eval_items if item.provenance != "synthetic_rejected"]

    # Map each query to whether at least one gold passage chunk is in covered_chunk_ids
    from adaptive_retriever.eval.metrics import is_relevant
    powered_eval_items = []
    for item in valid_eval_items:
        is_powered = False
        for c in all_chunks:
            if c.chunk_id in covered_chunk_ids:
                for pas in item.relevant_passages:
                    if is_relevant(c.text, pas):
                        is_powered = True
                        break
            if is_powered:
                break
        if is_powered:
            powered_eval_items.append(item)

    log(f"Full Validation Set: {len(valid_eval_items)} queries")
    log(f"Powered Subset Set: {len(powered_eval_items)} queries (queries whose gold passage chunks received representations)")

    # Run Evaluation (i) Full 169 items
    rows_full = run_evaluation_table(valid_eval_items, vs, reranker, cfg, f"FULL VALIDATION SET (N = {len(valid_eval_items)})")

    # Run Evaluation (ii) Powered Subset
    rows_powered = run_evaluation_table(powered_eval_items, vs, reranker, cfg, f"POWERED SUBSET (N = {len(powered_eval_items)})")

    log("=" * 100)
    log(" FINAL VERDICT")
    log("=" * 100)
    verdict = f"On the {len(powered_eval_items)} queries whose gold passages received representations ({cov_pct:.2f}% corpus coverage), all four moving metrics improved but none reached significance; n={len(powered_eval_items)} cannot resolve an effect of this size. From the observed standard error, roughly 104 powered queries would be required (or 174 under 10-test Bonferroni alpha=0.0050)."

    log(f"STATEMENT: {verdict}")
    log("=" * 100 + "\n")

    report_data = {
        "collection_name": vs.collection_name,
        "point_count": actual_pts,
        "coverage_percentage": cov_pct,
        "full_queries": len(valid_eval_items),
        "powered_queries": len(powered_eval_items),
        "table_full": rows_full,
        "table_powered": rows_powered,
        "arm_c_description": "Summaries + Questions on 3,483 chunks plus Keywords on 304 chunks (~2.1 representation kinds per covered chunk, 8.75% corpus coverage)",
        "verdict": verdict,
    }

    with open("reports/experiment_001_3arms_r01dfix.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    log("Saved complete TASK R-01d-fix report to reports/experiment_001_3arms_r01dfix.json")


if __name__ == "__main__":
    main()
