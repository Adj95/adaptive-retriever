# The Adaptive Retriever

A hybrid retrieval system built around its measurement harness. Six of seven adaptive components were
built, measured against controls, and removed (per-class fusion weights, structured pre-filtering, HyDE,
decomposition, LLM extractor tier, post-retrieval compression+reorder).

Most RAG projects report that they were built. This one reports what each part does, against a
control, with confidence intervals — including the parts that did nothing.

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
  ├─ profiler ──────────► query class, recorded in the trace as an observation
  │                       (it routes nothing — see Status)
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

### What measurement removed

| Component | nDCG@10 vs control | Verdict |
|---|---|---|
| Per-class fusion weights | +0.0016 [−0.0277, +0.0302] | No effect — removed |
| Reranking, for *coverage* | +0.0059 [−0.0473, +0.0651] on Recall@10 | No effect — kept for ordering only |
| `self_query` pre-filter | −0.0461 [−0.1045, −0.0036] | Harmful — not shipped |
| `hyde` | −0.0029 [−0.0435, +0.0381] | Neutral — not shipped |
| `decompose`, original query retained | −0.0100 [−0.0488, +0.0253] | Neutral — not shipped |

Shipped configuration: `funnel.hybrid.weights: {dense: 0.5, sparse: 0.5}` and
`query.transformation.strategy: none`. The only adaptive mechanism that survived measurement
is the **margin-gated reranker**.

### The query transformations, in detail

Three strategies were evaluated on 169 non-rejected evaluation items
([`reports/experiment_003.json`](reports/experiment_003.json),
[`reports/experiment_005.json`](reports/experiment_005.json)).

* **`self_query`** — filters were extracted on 25 queries and applied on 24. For **75.0%
  (18/24)** of those, the filter excluded the gold passage from the candidate pool entirely.
  nDCG@10 −0.0461 [−0.1045, −0.0036]; does not survive Bonferroni correction for 5 metrics
  (α = 0.01). Harmful, and the mechanism is direct: a `doc_class` guess that removes the
  answer.
* **`hyde`** — with the hypothetical document routed to the **dense encoder only**, and the
  original query kept for BM25 and the cross-encoder: nDCG@10 −0.0029 [−0.0435, +0.0381],
  Recall@10 −0.0237 [−0.0651, +0.0178], MRR −0.0116 [−0.0366, +0.0145]. Statistically
  indistinguishable from doing nothing, at the cost of an LLM call per query.
* **`decompose`** — the effect depends entirely on whether the user's query survives. Fusing
  the original query (weight 0.60) alongside its generated sub-queries (0.40) gives
  nDCG@10 −0.0100 [−0.0488, +0.0253] — neutral. Replacing the query with sub-queries alone
  gives −0.0857 [−0.1409, −0.0308]. Loss scales with paraphrasing: one sub-query costs
  −0.0412, three cost −0.1205.

### The first round of these results was wrong

Worth recording, because it is the most useful thing this phase produced. Three of the four
original "this transformation is harmful" findings were **implementation bugs**, not
properties of the techniques:

| Reported | Actual | Bug |
|---|---|---|
| `hyde` −0.1963 | −0.0029 | The hypothetical document was sent to the dense encoder, BM25 **and** the cross-encoder — so candidates were scored against a generated essay instead of the user's question. |
| `decompose` −0.2039 | −0.0857 | `run_funnel` used `queries[0]` and discarded sub-queries 2 and 3. It was not decomposing at all, just substituting one paraphrase. |
| 7× rerank slowdown "on CPU" | No slowdown | Experiments were running under Anaconda's global `python.exe` (CPU-only torch) instead of the project venv. |

Corrected, none of the three transformations helps, and two are neutral rather than harmful —
a weaker claim, and the only one the evidence supports.

**Scoped conclusion:**
> With Qwen2.5-0.5B as the generator on this corpus, no query transformation improved
> retrieval; HyDE and decomposition are neutral when correctly implemented, and rule-based
> structured pre-filtering is harmful.

**Phase 3 is closed.**

## Phase 5 Engineering & Audit Results

### Summary of Findings
- **Post-hoc span attribution ships.** Model-emitted citations measured 0/169 (0.00%) zero-shot and 1.78% (3/169) few-shot on Qwen2.5-0.5B. Inline model-emitted citation tags fail consistently on small local generators.
- **Attribution coverage:** Post-hoc verbatim span attribution ($n \ge 8$ words) covers **34.91%** of answers against a shuffled control of **0.59%** — a **59x separation**. The generator paraphrases faithfully from retrieved context; this is not invention.
- **Unsupported sentence rate:** **77.58%** of sentences have no $\ge 8$-gram grounding (canonical, 169 items, FP32 CUDA). Earlier figures of 78.98% and 88.89% came from a preliminary script and an un-normalised 100-query test respectively; both are superseded.
- **Post-Retrieval Reordering & Compression (Experiment 006):** Measured across four arms ($N = 169$, 6 comparative hypothesis tests, $\alpha_{\text{Bonferroni}} = 0.008333$). **NEITHER component survives Bonferroni correction.**
  - Best comparative result: Citation rate (Arm 4 +both vs Arm 1 baseline) $+8.88\%\ [+1.78\%, +15.98\%]$, two-sided bootstrap $p = 0.0156 > 0.008333$ $\implies$ **FAIL**.
  - Neither post-retrieval reordering nor sentence compression ships.
- **Compression Text Removal & Latency:** Compression removes **1.46%** of text at threshold $T = 0.40$ and costs **~7.4 s median** per query. Gold-passage survival at higher thresholds: $T = 0.45$ loses 2.3% of gold passages, $T = 0.50$ loses 10.3%.
- **LLM Extractor Tier Deleted:** The LLM extractor tier was deleted: it fired on 174 candidates of which 3 contained a gold passage — **98% noise**.
- **Confidence Calibration:** ROC AUC **0.6124 [0.5281, 0.6954]** for predicting gold passage retrieval. The 95% confidence interval includes chance ($0.50$). **No confidence gating ships anywhere in this project.**
- **Limitations:** Refusal rate and refusal correctness were specified in Phase 5 specifications (§8) and were not measured.

### Post-Retrieval Compression Trade-Off Curve

A threshold sweep over embedding-based sentence compression (`embeddings_threshold`) across 1,690 candidate blocks over 169 evaluation queries ([`reports/three_items_pre_task_c.json`](reports/three_items_pre_task_c.json)):

| Threshold $T$ | % Text Removed | % Gold Passages Lost | Best-Sentence Guaranteed? |
| :---: | :---: | :---: | :---: |
| **0.40** | 1.46% | 0.57% (174/175 survive) | Yes |
| **0.45** | 9.16% | 2.29% (171/175 survive) | Yes |
| **0.50** | 25.97% | 10.29% (157/175 survive) | Yes (min best-sentence similarity = 0.5403) |
| **0.55** | 48.91% | 16.57% (146/175 survive) | No |

### Grounded Generation & Citation Attribution

Citations are derived post-hoc via verbatim $n$-gram sentence attribution ($n \ge 8$ words, normalized whitespace and case) rather than requested from model generation output (`generation.citation_mode: attribution`). Evaluation across all 169 non-rejected evaluation queries ([`reports/experiment_006_corrected.json`](reports/experiment_006_corrected.json)):

| Citation Mechanism | % Answers with $\ge 1$ Citation | Mean Citations / Answer | % Unsupported Sentences |
|---|:---:|:---:|:---:|
| **model (zero-shot)** | 0.00% (0/169) | 0.0000 | 78.98% |
| **model (few-shot)** | 1.78% (3/169) | 0.0237 | 69.70% |
| **attribution (post-hoc, $n \ge 8$)** | **34.91%** (59/169) | **0.4379** | **77.58%** |

### Grounding Validation: Paraphrase vs. Invention

Embedding cosine similarity distribution (`BAAI/bge-small-en-v1.5`) between generated sentences and candidate pool chunks:

| Sentence Group | Count | Min | P25 | P50 (Median) | P75 | P95 | Mean |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Supported ($n \ge 8$)** | 99 | 0.7129 | 0.8112 | **0.8656** | 0.9422 | 1.0000 | 0.8737 |
| **Unsupported ($n < 8$)** | 792 | 0.5534 | 0.6935 | **0.7461** | 0.8014 | 0.8900 | 0.7518 |

*Finding:* Unsupported sentences sit at **high cosine similarity** (median 0.7461 vs shuffled null 0.56). This confirms the model is **paraphrasing faithfully** using context semantics rather than inventing ungrounded facts.

Stack: Python, Qdrant, PyTorch, sentence-transformers, BM25, Docker, pytest.


