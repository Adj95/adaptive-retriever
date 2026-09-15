# The Adaptive Retriever

A retrieval system that classifies each query and adapts its retrieval strategy to it,
measured against a fixed-strategy baseline on the same index.

Most RAG projects report that they were built. This one reports what it does, against a
control, with confidence intervals — including where it does nothing.

---

## Result

Adaptive funnel vs a fixed dense-retrieval baseline. Same corpus, same chunks, same
embedding model, same index, same 169 evaluation queries. The only difference is
adaptivity.

| Metric | Fixed baseline | Adaptive | Difference [95% CI] |
|---|---|---|---|
| Recall@5 | 0.3343 | **0.4497** | +0.1154 [+0.0561, +0.1775] |
| nDCG@5 | 0.3741 | **0.5131** | +0.1391 [+0.0642, +0.2168] |
| Recall@10 | 0.3817 | **0.4793** | +0.0976 [+0.0443, +0.1509] |
| nDCG@10 | 0.4344 | **0.5890** | +0.1546 [+0.0742, +0.2389] |
| MRR | 0.2607 | **0.3483** | +0.0877 [+0.0378, +0.1371] |

Confidence intervals come from a **paired** bootstrap resampling queries, 1000 resamples,
seed 42. All five intervals exclude zero. Full output: [`reports/experiment_001.json`](reports/experiment_001.json).

Latency figures are pending re-measurement with the execution device verified.


---

## The finding that surprised me

A threshold sweep over the rerank gate ([`reports/experiment_002.json`](reports/experiment_002.json))
shows that **cross-encoder reranking improves ordering but not coverage.**

| Gate margin | Fire rate | Recall@10 vs never-rerank | nDCG@10 vs never-rerank |
|---|---|---|---|
| 0.00 | 0% | — | — |
| 0.15 | 18.3% | +0.0000 [-0.0178, +0.0178] | +0.0064 [-0.0199, +0.0293] |
| 0.30 | 77.5% | +0.0178 [-0.0237, +0.0592] | +0.0779 [+0.0151, +0.1427] |
| 1.00 | 100% | +0.0059 [-0.0473, +0.0651] | +0.0737 [+0.0063, +0.1450] |

Recall@10 is statistically flat across the entire sweep — reranking every query instead of
none changes coverage by `+0.0059 [-0.0473, +0.0651]`. nDCG and MRR do move.

So the entire `0.3817 → 0.4793` coverage gain in the headline table comes from **hybrid
multi-representation retrieval**, not from the reranker. The reranker only reorders what
retrieval already found. That is what a cross-encoder should do, and now it is measured
rather than assumed.

---

## How it works

```
query
  │
  ├─ profiler ──────────► one of 6 query classes + extracted filters
  │
  ├─ stage 1  structured pre-filter    22,422 → N   (indexed payload fields only)
  ├─ stage 2  hybrid retrieval             N → 100  (dense + BM25, weighted RRF)
  └─ stage 3  gated cross-encoder        100 → 10   (fires only when ranking is uncertain)
```

**Multi-representation indexing.** Every chunk is indexed several times — its raw text plus
LLM-generated summaries, questions and keywords, under a value-based tiering policy that
only spends generation on chunks dense enough to warrant it. 12,074 chunks become 22,422
indexed vectors.

**Representations collapse to chunks *before* fusion, not after.** This is the load-bearing
detail. If you fuse at representation level and merge afterwards, a chunk with four
representations accumulates four RRF contributions while a chunk with one accumulates one —
so chunks get ranked partly by how many times they happen to be indexed, which has nothing
to do with relevance. `tests/test_rrf_fusion.py::test_representation_count_does_not_bias_ranking`
pins this and demonstrates that the naive alternative really does flip the order.

**Fusion weights.** Fusion uses fixed dense/sparse weightings ({dense: 0.5, sparse: 0.5}) for RRF across all queries.
Every weight pair is validated to sum to 1.0 so fused scores stay comparable —
otherwise the rerank gate margin would mean something different for each query.

**The rerank gate is relative, not absolute.** The gate fires when the score drop-off between
rank 1 and rank k is small, measured as `(top - cutoff) / top`. An absolute margin does not
work here: RRF scores cap at `1/(rrf_k+1) = 0.0164`, so any threshold above that is always
open. That bug shipped once and the sweep is what caught it.

**Content-addressed index.** The collection name derives from a hash covering only what
changes the *contents* of the index — not storage location, device, batch size or dtype.
Moving from embedded Qdrant to a server, or from CPU to GPU, does not trigger a re-index.

---

## Honest caveats

These belong in the README, not buried:

- The evaluation set is **LLM-generated**. Of 197 items, 31 are human-verified, 138 are
  unverified synthetic, and 28 were excluded.
- The 28 excluded items were **provably unanswerable**: the generator had paired each
  multi-hop query with one relevant passage and one randomly sampled one. They carry a
  recorded rejection reason rather than being silently dropped.
- A further 24 items previously marked verified were **downgraded** after their review
  timestamps showed intervals averaging 1.25 seconds — too fast to constitute review.
- The gate-threshold sweep is **exploratory**. It ran 18 paired comparisons (6 margins ×
  3 metrics) at 95%; under Bonferroni correction none of them remains significant.
- The gate-threshold sweep is **exploratory**. It ran 18 paired comparisons (6 margins ×
  3 metrics) at 95%; under Bonferroni correction none of them remains significant.
  `gate_score_margin: 0.30` was tuned under per-class weights since removed (which changed
  the RRF score distribution; under uniform weights the fire rate shifted from 77.5% to 55.6%,
  94/169 queries, with relative margin distribution p50 = 0.2812, p95 = 0.4911, mean = 0.2951).
  The threshold is therefore provisional.
- **Transformation Latency**: Latency figures are pending re-measurement with the verified execution device.
- Absolute scores are modest (Recall@10 ≈ 0.48). The claim here is the **controlled
  difference**, not state-of-the-art absolute retrieval.
- Everything runs on one consumer GPU (GTX 1650, 4 GB). fp32 is used rather than fp16
  because the card has no tensor cores and fp16 measured 3.5× slower.

---

## Engineering

- **117 tests**, mutation-verified: every retrieval behaviour is paired with a test proven
  to fail when that behaviour is removed. Each was confirmed by injecting the specific
  one-line change and watching the intended test fail.
- Every stage emits a **trace** to `reports/traces/` with input count, output count, elapsed
  time and the decision taken. A stage that skipped says so; it does not report `applied`.
- **Reproducibility**: Retrieval candidate sets before reranking are **100% bit-identical and reproducible** across execution environments; post-rerank candidate selection can shift by a small number of items (e.g., 2/169 queries) due to float32 non-determinism in cross-encoder batch composition and sequence padding.
- Offline LLM generation is cached and deterministic — 10,449 representations, seeded,
  reproducible.
- No silent fallbacks. Embedding model, sparse model and reranker all **raise** on load
  failure rather than returning stub vectors or constant scores. Both of those bugs shipped
  in an early version and are the reason the rule exists.

---

## Quickstart

```bash
# Qdrant server (payload indexes are a no-op in embedded mode)
docker run -p 6333:6333 -v $(pwd)/data/qdrant_server:/qdrant/storage qdrant/qdrant

pip install -e .
python -m adaptive_retriever.cli ingest
python -m adaptive_retriever.cli funnel "what methods were used to evaluate results" --trace
python -m pytest -q
```

Reproduce the experiments:

```bash
python scripts/run_experiment_001.py    # adaptive vs fixed baseline
python scripts/run_experiment_002.py    # rerank gate threshold sweep
python scripts/run_experiment_003.py    # self_query pre-filter evaluation (Task B)
python scripts/run_experiment_005.py    # HyDE and Decomposition evaluation (Experiment 005)
```

---

## Layout

```
src/adaptive_retriever/
  ingest/          parsing, chunking, representation generation
  funnel/          fusion, prefilter, rerank, pipeline, baseline
  query/           query profiler, transformation interface
  eval/            metrics, bootstrap CIs, eval-set construction
  vectorstore.py   Qdrant, hybrid search, raw-text hydration
reports/           experiment output and stage traces
```

---

## Status

**Phases 1, 2, and 3 Complete.**

Query transformation strategies (`self_query`, `hyde`, `decompose`) were systematically evaluated on 169 non-rejected evaluation items and **measured-and-rejected**:

* **`self_query`**: Pre-filters were extracted on 25 queries and applied on 24 queries. For **75.0% (18/24)** of filtered queries, the pre-filter excluded the gold passage from the candidate pool entirely. The observed nDCG@10 difference of −0.0461 [95% CI −0.1045, −0.0036] does not survive Bonferroni correction for 5 metrics ($\alpha = 0.01$).
* **`hyde`**: Hypothetical document generation degraded retrieval ($\text{nDCG}@10$ diff **−0.1963** [95% CI −0.2891, −0.1132], surviving Bonferroni $\alpha = 0.005$). Latency figures are pending re-measurement with the verified execution device.
* **`decompose`**: Query decomposition degraded retrieval ($\text{nDCG}@10$ diff **−0.2039** [95% CI −0.2910, −0.1306], surviving Bonferroni $\alpha = 0.005$). Mean sub-queries: **2.35 / query** (distribution: 1: 40, 2: 30, 3: 99). Degenerate sub-queries rejected: 2 total. Latency figures are pending re-measurement with the verified execution device.

**Scoped Conclusion**:
> *HyDE and decomposition, implemented with Qwen2.5-0.5B, degrade retrieval on this corpus (nDCG@10 −0.196 and −0.204, both surviving Bonferroni at $\alpha = 0.005$).*

Shipped configuration: `query.transformation.strategy: none`. **Phase 3 is closed.**

Stack: Python, Qdrant, PyTorch, sentence-transformers, BM25, Docker, pytest.
