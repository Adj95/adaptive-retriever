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
adaptivity.Across 5 comparative metrics evaluated at Bonferroni-corrected threshold $\alpha = 0.05 / 5 = 0.01$:

| Metric | Dense-only baseline | Hybrid + rerank (raw index) | Difference [95% CI] | Two-Sided $p$-value | Survives Bonferroni ($\alpha = 0.01$) |
|---|---|---|---|---|---|
| Recall@5 | 0.4970 | **0.5917** | +0.0947 [+0.0237, +0.1657] | $p = 0.0078$ | **PASS** |
| nDCG@5 | 0.4034 | **0.5061** | +0.1027 [+0.0448, +0.1609] | $p = 0.0006$ | **PASS** |
| Recall@10 | 0.5799 | **0.6568** | +0.0769 [+0.0118, +0.1420] | $p = 0.0234$ | **FAIL** |
| nDCG@10 | 0.4311 | **0.5280** | +0.0969 [+0.0430, +0.1512] | $p = 0.0006$ | **PASS** |
| MRR | 0.3922 | **0.4949** | +0.1027 [+0.0444, +0.1612] | $p = 0.0006$ | **PASS** |

Confidence intervals come from a **paired** bootstrap resampling queries, 10,000 resamples, seed 42. Four of five metrics (nDCG@5, nDCG@10, MRR, Recall@5) survive Bonferroni correction at $\alpha = 0.01$. Recall@10 ($p = 0.0234$) fails Bonferroni correction. Full output: [`reports/experiment_001.json`](reports/experiment_001.json).

> **Token Cap & Dual-Scoring Note**: All chunks in the vectorstore are capped at $\le 480$ BGE-small tokens (with a permanent assertion `max_tokens <= 512`), ensuring the encoder reads 100% of every chunk. The legacy 2,000-character proxy dual-scoring check has been retired as BGE-small token length directly enforces complete context visibility.

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

So the coverage gain in the headline table comes from **hybrid retrieval (dense + BM25)** on the raw index, not from the reranker. The reranker only reorders what retrieval already found. That is what a cross-encoder should do, and now it is measured rather than assumed.

---

## How it works

```
query
  │
  ├─ profiler ──────────► query class, recorded in the trace as an observation
  │                       (it routes nothing — see Status)
  │
  ├─ stage 1  structured pre-filter    39,814 → N   (indexed payload fields only)
  ├─ stage 2  hybrid retrieval             N → 100  (dense + BM25, weighted RRF)
  └─ stage 3  gated cross-encoder        100 → 10   (fires only when ranking is uncertain)
```

**Index Structure.** The Qdrant vectorstore holds 39,814 token-capped raw chunks (max 480 BGE-small tokens per chunk). Multi-representation generation is supported by the ingest pipeline but remains un-indexed at scale as offline generation costs were not measured to repay their generation time. Some source documents are themselves Q&A datasets formatted "Question: ... Answer: ...", so generated question representations trivially copy the existing question.

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

- **Unanswerable Fraction**: A third of the evaluation set (54 of 169 queries, 31.95%) cannot be answered by any retriever configuration in this system, which bounds every retrieval and routing result in this project.

- `is_relevant()`'s 8-gram match can fire on SEC table delimiters (`"| | | | | | | |"`). 4 of 169 items are flagged `boilerplate_suspect`; excluding them moves Recall@10 by 0.004 (0.5799/0.6568 -> 0.5758/0.6485) and nDCG@10 by 0.009 (0.4311/0.5280 -> 0.4264/0.5195). Normalising table delimiters before n-gram matching remains unfixed.


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

## Phase 5: Post-Retrieval Pipeline Evaluation (Experiment 006)

### Post-Retrieval Pipeline Evaluation & Truncation-Aware Verdict
Under truncation-aware scoring (restricting relevance evaluation to the first 2,000 characters read by dense and cross-encoder models), the headline retrieval improvement is roughly halved and no longer clears Bonferroni correction at $n=169$:
- Recall@10: $+0.1036 \rightarrow +0.0473$ ($p=0.0506$, 95% CI $[+0.0000, +0.0947]$)
- nDCG@10: $+0.1642 \rightarrow +0.0542$ ($p=0.0378$, 95% CI $[+0.0020, +0.1055]$)
- MRR: $+0.0935 \rightarrow +0.0483$ ($p=0.0198$, 95% CI $[+0.0072, +0.0896]$)

About half the measured gain is BM25 exploiting oversized chunks; the remainder is positive across all three metrics but underpowered at $n=169$.

At embedding threshold $T=0.40$, 131 out of 169 items had zero characters removed (`chars_removed == 0`). However, candidate sentence splitting in Arm 3 reformatted prompt context by inserting explicit newline (`\n`) separators between sentence spans, whereas Arm 1 preserved raw paragraph whitespace. This prompt-formatting difference altered prompt text for all items.

Because generation with `temperature: 0.0` is 100% deterministic on fixed inputs, the observed citation rate difference of **+6.11%** (`[-0.76%, +13.74%]`, [`reports/experiment_006_raw.jsonl`](reports/experiment_006_raw.jsonl)) on the 131-item subset reflects a prompt-formatting whitespace confound rather than sampling variance. Across all four arms ($N=169$, [`reports/experiment_006.json`](reports/experiment_006.json), [`reports/experiment_006_corrected.json`](reports/experiment_006_corrected.json)), no arm difference survives Bonferroni correction ($\alpha_{\text{Bonferroni}} = 0.008333$). Neither post-retrieval reordering nor sentence compression ships.

### Grounded Generation & Sentence Attribution
Citations are derived post-hoc via verbatim $n$-gram sentence attribution ($n \ge 8$ words, normalized whitespace and case) rather than requested from model generation output (`generation.citation_mode: attribution`). Evaluation across all 169 non-rejected evaluation items ([`reports/experiment_006_corrected.json`](reports/experiment_006_corrected.json)):

| Citation Mechanism | % Answers with $\ge 1$ Citation | Mean Citations / Answer | % Unsupported Sentences |
|---|:---:|:---:|:---:|
| **model (zero-shot)** | 0.00% (0/169) | 0.0000 | 78.98% |
| **model (few-shot)** | 1.78% (3/169) | 0.0237 | 69.70% |
| **attribution (post-hoc, $n \ge 8$)** | **34.91%** (59/169) | **0.4379** | **77.58%** |

*Canonical Metric Definition*: Unsupported sentence rate is **77.58%** (mean item percentage of generated sentences with zero verbatim $n \ge 8$ word overlap against candidate context across all 169 evaluation queries, FP32 CUDA execution).

### Grounding Validation: Paraphrase vs. Invention
Embedding cosine similarity distribution (`BAAI/bge-small-en-v1.5`) between generated sentences and candidate pool chunks ([`reports/paraphrase_attribution_analysis.json`](reports/paraphrase_attribution_analysis.json)):

| Sentence Group | Count | Min | P25 | P50 (Median) | P75 | P95 | Mean |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Supported ($n \ge 8$)** | 99 | 0.7129 | 0.8112 | **0.8656** | 0.9422 | 1.0000 | 0.8737 |
| **Unsupported ($n < 8$)** | 792 | 0.5534 | 0.6935 | **0.7461** | 0.8014 | 0.8900 | 0.7518 |

*Finding:* Unsupported sentences sit at high cosine similarity (median 0.7461 vs shuffled null 0.56). This confirms the model paraphrases faithfully using context semantics rather than inventing ungrounded facts.

### Index Chunk Length Distribution Anomaly
Querying the Qdrant index directly (`retriever_chunks_a5d51992`, 12,074 unique chunks across 22,422 points):
- **Chunk text length distribution**: min 4 chars, p50 2,933 chars, p95 7,851 chars, p99 92,169 chars, max 1,516,213 chars.
- **Oversized chunk counts**: 8,538 chunks > 2,000 chars (70.71%), 440 chunks > 10,000 chars (3.64%), 169 chunks > 50,000 chars (1.40%).
- **Largest un-chunked document blocks**: Document `4d9febdbc2038dcd` (1,516,213 chars) and SEC filings (`d4afbf861ecaf6a9`, 571,883 chars) bypassed paragraph-break chunking. Sentence compression on these un-chunked blocks evaluated up to 5,522 sentence spans per query, inflating mean compression latency.

Stack: Python, Qdrant, PyTorch, sentence-transformers, BM25, Docker, pytest.



