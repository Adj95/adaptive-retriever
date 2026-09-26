# The Adaptive Retriever

## What this is
A hybrid retrieval system in which every optional component is gated behind a measurement on the target corpus. Rather than assuming that query expansion, hypothetical document embeddings, structured pre-filtering, or cascade routing automatically improve retrieval performance, each component is evaluated against a fixed baseline and control on the specific dataset. Gains are modest and strongly corpus-dependent: a technique that yields measurable improvements on one domain may yield neutral or negative results on another. Consequently, optional components ship disabled by default and are enabled only when supported by empirical evidence on the target corpus.

## Quickstart

To run the system from a fresh clone:

```bash
git clone https://github.com/Adj95/adaptive-retriever.git
cd adaptive-retriever
python -m venv .venv
.venv\Scripts\activate          # Windows (.venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
docker start qdrant             # Start Qdrant server on http://localhost:6333
python scripts/ingest_beir_corpora.py --corpus nfcorpus
python scripts/demo.py
```

### What it looks like

```
QUERY     Why Deep Fried Foods May Cause Cancer
          (NFCorpus test query PLAIN-1, 0 gold documents)

RETRIEVED top 5 of 10
  1   MED-2418_c0000   score 7.015    gold: no
  2   MED-4070_c0000   score 3.865    gold: no
  3   MED-5269_c0001   score 0.060    gold: no
  4   MED-4818_c0000   score -0.132   gold: no
  5   MED-5033_c0000   score -0.470   gold: no
  rerank fired: yes (top-1 margin 0.28 < 0.30)

ANSWER
  Deep fried foods may cause cancer because they contain high levels of
  harmful substances like aldehydes, acrolein, heterocyclic amines, polycyclic
  aromatic hydrocarbons, and acrylamide. These compounds can form potentially
  carcinogenic agents when heated, which increases the risk of developing
  cancer. Additionally, studies show that frying meat at higher temperatures
  can lead to the formation of certain chemicals that can damage cells and
  contribute to cancer development.

GROUNDING 0 of 3 sentences matched >= 8-gram verbatim to retrieved context
  sentence 1  ungrounded
  sentence 2  ungrounded
  sentence 3  ungrounded

COST      retrieval 6,033 ms | rerank 2,133 ms | generation 6,771 ms | total 16,554 ms
```

### When the generator gets it wrong

The 0.5B model produced a self-contradicting answer ("Yes... statins decrease the risk") and the grounding engine correctly flagged all 3 sentences as ungrounded (0 of 3 matched).

```
QUERY     Do cholesterol statin drugs cause breast cancer?
          (NFCorpus test query PLAIN-2, 24 gold documents)

RETRIEVED top 5 of 10
  1   MED-14_c0000     score 4.554    gold: yes
  2   MED-2429_c0000   score 4.097    gold: yes
  3   MED-10_c0000     score 3.304    gold: yes
  4   MED-2431_c0000   score 3.289    gold: yes
  5   MED-4827_c0000   score 2.246    gold: no
  rerank fired: yes (top-1 margin 0.19 < 0.30)

ANSWER
  Yes, cholesterol statin drugs do cause breast cancer. According to the
  context provided, "Emerging evidence suggests that statins' may decrease the
  risk of cancers." However, the study does not provide specific information
  about the link between statin use and breast cancer. Therefore, it's
  insufficient to determine if cholesterol statin drugs cause breast cancer
  based solely on the given information.

GROUNDING 0 of 3 sentences matched >= 8-gram verbatim to retrieved context
  sentence 1  ungrounded
  sentence 2  ungrounded
  sentence 3  ungrounded

COST      retrieval 5,882 ms | rerank 1,803 ms | generation 5,138 ms | total 18,918 ms
```

## Results
Two components cleared their controls — hybrid fusion and cross-encoder reranking — and nine did not.

Three corpora were evaluated using nDCG@10 as the primary comparable metric across fixed dense baselines (`BAAI/bge-small-en-v1.5`) and hybrid dense+sparse Reciprocal Rank Fusion (RRF) with gated cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`):

| Corpus | Description | Sample Size ($n$) | Fixed Dense Baseline | Hybrid + Rerank | Difference | Verdict ($\alpha_{\text{Bonferroni}} = 0.01$) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Corpus A** | 428 docs, synthetic eval | 169 | 0.4311 | 0.5280 | +0.0969 | **PASS** |
| **NFCorpus** | BEIR benchmark, human qrels | 323 | 0.3530 | 0.3667 | +0.0137 | **FAIL** |
| **HotpotQA** | BEIR 20k random subsample | 500 | 0.9201 | 0.9451 | +0.0250 | **PASS** |

The hybrid retrieval advantage is strongly corpus-dependent: the measured nDCG@10 gain falls by roughly $7\times$ on NFCorpus (+0.0137) compared to Corpus A (+0.0969), where query relevance judgments are human-annotated rather than synthetically generated. For all three corpora, statistical significance was evaluated against a Bonferroni-corrected threshold of $\alpha = 0.01$ (5 metric comparisons at global $\alpha = 0.05$). Note that the HotpotQA evaluation uses a 20k document random subsample with easy negatives and is not directly comparable to published figures on full benchmark suites.

## External validation
On NFCorpus, the dense-only baseline achieved an nDCG@10 of 0.3530 ([`reports/r06b_baseline.json`](reports/r06b_baseline.json)), comparing closely against the published BGE-small-en-v1.5 figure of approximately 0.34 on the BEIR benchmark (Thakur et al., 2021, *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* / BAAI BGE Model Card). This confirms that the measurement harness reproduces expected baseline retrieval behavior within standard variation attributable to document chunking and indexing parameters. However, these figures are not directly comparable to published BEIR benchmark scores because this system evaluates retrieval over chunked document segments, whereas standard BEIR evaluations process un-chunked full documents.

## What ships
The core architecture ships with the following components verified and enabled:
- **Hybrid Dense + Sparse RRF with Collapse-Before-Fusion**: Dense vector search (`BAAI/bge-small-en-v1.5`) combined with BM25 sparse retrieval via Reciprocal Rank Fusion, collapsing multiple chunk representations to their parent chunk prior to score aggregation to eliminate indexing frequency bias.
- **Cross-Encoder Reranking**: Cross-encoder reranking cleared its control on ordering (nDCG@10 +0.0737, 95% CI [+0.0063, +0.1450], excludes zero) but not on coverage (Recall@10 +0.0059, 95% CI [-0.0473, +0.0651], includes zero).
- **Post-Hoc Span Attribution**: Deterministic verbatim $n$-gram ($n \ge 8$ words) sentence attribution to trace generated answer spans back to retrieved candidate context.
- **Qdrant Server Integration**: Server-mode vector store with 7 indexed payload fields (`doc_class`, `contains_table`, `entities`, `date_refs`, `access_tags`, `kind`, `tier`).
- **Evaluation Harness**: Statistical evaluation pipeline including paired bootstrap resampling (10,000 iterations), shuffled controls, Bonferroni multiplicity correction, and automated report verification via `scripts/verify_report.py`.

## What does not ship, and why
All optional adaptive components measured in this project failed to clear their controls under target evaluation conditions and do not ship in the default pipeline configuration:

| Component | Measured Effect | Primary Report Source |
|---|---|---|
| **per-class fusion weights** | +0.0000 on Recall@10 | [`reports/experiment_001.json`](reports/experiment_001.json) |
| **self-query prefiltering** | -0.0461 on Corpus A; NOT APPLICABLE on NFCorpus (no metadata schema) - never fairly tested | [`reports/experiment_003.json`](reports/experiment_003.json), [`reports/r06c1_transformations.json`](reports/r06c1_transformations.json) |
| **HyDE** | -0.0029 (Corpus A) / -0.0026 (NFCorpus) | [`reports/experiment_005.json`](reports/experiment_005.json), [`reports/r06c1_transformations.json`](reports/r06c1_transformations.json) |
| **query decomposition** | -0.0100 (Corpus A) / +0.0012 (NFCorpus) | [`reports/experiment_005.json`](reports/experiment_005.json), [`reports/r06c1_transformations.json`](reports/r06c1_transformations.json), [`reports/r06d.json`](reports/r06d.json) |
| **LLM extractor tier** | 174 candidates fired, 3 held gold | [`reports/three_items_pre_task_c.json`](reports/three_items_pre_task_c.json) |
| **compression + reorder** | failed Bonferroni ($\alpha = 0.004167$); difference traced to a whitespace confound | [`reports/experiment_006_corrected.json`](reports/experiment_006_corrected.json) |
| **multi-representation indexing** | +0.0044, $p = 0.1592$ at 41.8% coverage on NFCorpus; no resolvable effect on Corpus A at 8.75% | [`reports/r06c2_multirep.json`](reports/r06c2_multirep.json), [`reports/r03_tiers.json`](reports/r03_tiers.json) |
| **cascade router** | cancelled before implementation - three reasons, see below | [`reports/r02f_oracle.json`](reports/r02f_oracle.json) |

Every component implementation remains fully intact in the codebase behind configurable flags for target domain experimentation.

## Key findings
1. **HyDE raises similarity indiscriminately** ([`reports/r06d.json`](reports/r06d.json), [`reports/r06c1_transformations.json`](reports/r06c1_transformations.json)): Hypothetical generated passages sit +0.0134 closer in cosine space to gold documents than raw queries (0.6359 vs 0.6225), but sit +0.0205 closer to distractor documents (0.7018 vs 0.6813). Because similarity to noise improves more than similarity to signal, relative candidate ranking remains unchanged. Manual quality audit confirmed 83.6% of generated passages were coherent clinical prose, demonstrating that this is not a generation-quality failure.
2. **Retrieval-tier routing cannot pay for itself** ([`reports/r02f_oracle.json`](reports/r02f_oracle.json), [`reports/r02d.json`](reports/r02d.json), [`reports/r02b_gate.json`](reports/r02b_gate.json)): Retrieval accounts for only 1.0% to 4.5% of end-to-end query latency (75.76 ms to 303.32 ms out of 803.16 ms to 6,772.14 ms total); answer generation accounts for the remaining 95.5%+. Tiering retrieval alone yields a net latency savings of -0.51%. An extractive tier (Tier 0) is 66x cheaper than Tier 2 (75.76 ms vs 5,021.43 ms) but contains the gold answer on only 37.87% of queries (64/169), capping a perfect oracle router at a maximum cost reduction of 36.06%. The best escalation gate signal reached ROC AUC 0.6124 [0.5281, 0.6954] — identically matching the confidence score previously measured and rejected.
3. **Half the original headline gain was a chunking artifact** ([`reports/experiment_006_corrected.json`](reports/experiment_006_corrected.json), [`reports/r01a_rebuild_stats.json`](reports/r01a_rebuild_stats.json)): A legacy chunker lacking a hard character-count fallback emitted whole documents as single chunks, resulting in 70.71% of the index (8,538 chunks) exceeding the dense encoder's 512-token window (~2,000 characters). Consequently, BM25 and the relevance evaluator evaluated text beyond character 2,000 that the dense embedder never saw. When relevance evaluation was restricted to the text actually read by the encoder, the headline nDCG@10 gain dropped from +0.1642 to +0.0542. The index was subsequently rebuilt with strict 480-token chunk caps.
4. **On NFCorpus, distractors are more similar to queries than gold documents** ([`reports/r06d.json`](reports/r06d.json)): Non-gold distractor documents exhibit higher average cosine similarity to raw queries than gold documents (0.6813 vs 0.6225). Under these corpus conditions, similarity-boosting query transformations amplify distractor retrieval rather than gold target extraction.
5. **Multi-representation indexing at partial coverage is unresolvable** ([`reports/r06c2_multirep.json`](reports/r06c2_multirep.json), [`reports/r03_tiers.json`](reports/r03_tiers.json)): At 41.82% corpus coverage on NFCorpus (1,872 high-tier chunks), multi-representation indexing yielded a non-significant nDCG@10 shift of +0.0044 ($p = 0.1592$). At 8.75% coverage on Corpus A (3,483 chunks), the effect was similarly unresolvable. Partial-coverage multi-representation indexing did not demonstrate statistically resolvable benefit or harm under these conditions.

## Next step: a calibration procedure

Components are currently enabled or disabled by hand, using the measurements in `reports/`. The natural next step is to make that decision automatic and reproducible for a new corpus.

The procedure would use fixed-sequence testing rather than Bonferroni correction:
1. Rank candidate components by expected ROI (or cost, ascending).
2. Test the first component against the baseline on a calibration split of the target corpus at $\alpha = 0.05$.
3. If it passes, keep it and test the next component against the new pipeline.
4. Stop at the first failure. Do not test remaining components.

Because testing stops at the first non-rejection, fixed-sequence testing controls family-wise error rate at $\alpha = 0.05$ across the entire pipeline without penalty for multiple comparisons. On a 9-component pool, every tested component gets the full $\alpha = 0.05$ budget rather than the Bonferroni-corrected $\alpha = 0.0056$ - a $9\times$ statistical power advantage for early components.

Two requirements make this procedure rigorous:
- **Pre-registered sequence**: Component test order must be committed before observing target corpus metrics.
- **Held-out validation split**: The final calibrated pipeline configuration must be evaluated on an untouched validation split. Reporting calibration-split numbers as system performance produces optimistic bias.

This procedure is not implemented in the current codebase; pipeline decisions in `reports/` were made using parallel evaluation with Bonferroni correction.

## Reproducibility

BEIR benchmark evaluations (NFCorpus and HotpotQA) are **fully reproducible** from public downloads. The primary proprietary corpus (Corpus A, 428 documents) is not included in this repository due to distribution constraints, but its raw per-query evaluation outputs and metrics are preserved in [`reports/`](reports/).

To reproduce BEIR evaluations from scratch:

```bash
# 1. Download and ingest NFCorpus
python scripts/ingest_beir_corpora.py --corpus nfcorpus
python scripts/eval_beir_baseline.py --corpus nfcorpus

# 2. Download and ingest HotpotQA (20k subsample)
python scripts/ingest_beir_corpora.py --corpus hotpotqa
python scripts/eval_beir_baseline.py --corpus hotpotqa
```

To verify report integrity and metric computations against pre-computed report JSON files:
```bash
python check_preflight.py
python scripts/verify_report.py reports/experiment_001.json
python scripts/verify_report.py reports/r06b_baseline.json
python scripts/verify_report.py reports/r06c1_transformations.json
python scripts/verify_report.py reports/r06c2_multirep.json
python scripts/verify_report.py reports/r06d.json
```

Verified Collection Hashes and Active Point Counts (Qdrant Server):
- Corpus A (Rebuilt): `retriever_chunks_2e6f1c8d` (47,034 points) — raw results in [`reports/`](reports/)
- NFCorpus Standard Index: `beir_nfcorpus_2e6f1c8d` (4,476 points) — fully reproducible
- NFCorpus Multi-Rep Index: `beir_nfcorpus_multirep_cb65e1ba` (10,046 points) — fully reproducible
- HotpotQA 20k Subsample Index: `beir_hotpotqa_2e6f1c8d` (20,002 points) — fully reproducible
- HotpotQA 100k Subsample Index: `beir_hotpotqa_100k_2e6f1c8d` (100,018 points) — fully reproducible

*Note on Legacy Index*: The legacy pre-rebuild collection `retriever_chunks_a5d51992` was dropped during the R-01a-fix2 rebuild and no longer exists in the Qdrant vector store server.

## Limitations
- **Synthetic Evaluation Set Integrity**: Corpus A's evaluation set is synthetic ($n = 169$), generated by an LLM that was caught assigning query class labels by rota and pairing multi-hop queries with random passages. Both defects were audited and rejected queries excluded ([`reports/r01e_analysis.json`](reports/r01e_analysis.json)).
- **Unanswerable Query Ceiling**: 31.95% of Corpus A evaluation queries (54 of 169) are unanswerable by any retriever tier in this system, capping maximum achievable retrieval success at 68.05% ([`reports/r02f_oracle.json`](reports/r02f_oracle.json)).
- **Hardware & Model Scoping**: Offline generation and transformations were executed using `Qwen2.5-0.5B-Instruct` in FP32 precision on an NVIDIA GTX 1650 GPU ([`reports/r06c1_transformations.json`](reports/r06c1_transformations.json)). All measured nulls are scoped strictly to this generator setup; HyDE performance with larger LLM architectures was not evaluated.
- **NFCorpus Recall Ceiling**: NFCorpus query recall is capped by high gold-document density (median $p50 = 16$ gold documents per query), limiting top-$k$ recall ceilings ([`reports/r06b_baseline.json`](reports/r06b_baseline.json)).
- **Subsampled Benchmark Headroom**: HotpotQA evaluation uses a 500-query random subsample with easy negative distractors. Baseline dense retrieval hits an nDCG@10 ceiling of 0.9201; expanding the random distractor pool $5\times$ (from 20k to 100k) did not lower the ceiling ([`reports/r06c_prep.json`](reports/r06c_prep.json)).
- **Relevance Match Boilerplate**: The `is_relevant()` matching function can trigger on repeated table delimiters (e.g., `"| | | | |"`), flagging Corpus A items with boilerplate table rows.
- **Pre-Rebuild Index Scope**: All Phase 5 post-retrieval pipeline measurements were conducted on the legacy pre-rebuild index (collection `retriever_chunks_a5d51992`).
- **Prompt Token Context Discrepancy**: Measured prompt token counts indicate that the answer generator received approximately one-third of the context implied by tier configuration definitions ([`reports/r02c_gate.json`](reports/r02c_gate.json)).

## Architecture
The retrieval architecture consists of an ingest parser, Qdrant hybrid vector store, gated cross-encoder reranker, and verbatim sentence attribution engine. Detailed component wiring and pipeline stage parameters are defined in [`config/default.yaml`](config/default.yaml).

```
query
  │
  ├─ stage 1  structured pre-filter    (optional metadata payload filtering)
  ├─ stage 2  hybrid retrieval         (dense BGE-small + BM25 sparse, RRF fusion)
  └─ stage 3  gated cross-encoder      (ms-marco-MiniLM-L-6-v2 score-margin gating)
```
