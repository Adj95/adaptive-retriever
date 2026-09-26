# Portability and Component Deployment Protocol

This document outlines the protocol for porting and deploying this retrieval system on a new target corpus.

## Default Configuration Notice
The default pipeline configuration (`funnel.hybrid.weights: {dense: 0.5, sparse: 0.5}`, `query.transformation.strategy: none`, gated cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`) represents the baseline configuration empirically justified across our reference corpora. It is **not** a general recommendation or guarantee of optimality for unmeasured domains. Every optional component must be evaluated against target corpus controls prior to production enablement.

---

## Deployment Protocol

When deploying to a new domain or dataset:

1. **Ingest and Index Corpus**: Parse raw documents, apply strict token-window chunking ($\le 480$ tokens for 512-token encoders), and populate Qdrant vector payload indexes.
2. **Construct an Evaluation Dataset**:
   - Assemble a domain-specific evaluation set of $\ge 400$ queries with human-annotated relevance judgments (qrels), or adapt an established BEIR benchmark subset with gold relevance annotations.
   - Avoid unverified synthetic query sets that may exhibit periodicity, class rotas, or unanswerable query items.
3. **Run Component Measurement Suite**:
   - Evaluate candidate components (query transformations, multi-representation indexing, structured pre-filtering, tier routing) against fixed baseline controls.
   - Collect paired query metrics (nDCG@10, Recall@10, MRR) across 10,000 paired bootstrap resamples.
4. **Apply Multiplicity Correction**:
   - Calculate Bonferroni-corrected significance thresholds ($\alpha = 0.05 / M$ where $M$ is the number of comparative hypotheses evaluated).
   - Enable **only** those components whose measured performance gains clear their control after multiplicity correction.

---

## Reference Corpus Component Suitability Table

The table below summarizes measured performance across reference corpora and provides deployment guidance for when each component is likely to earn its place on a new corpus.

| Component | Reference Measurements | Primary Source Reports | When Component is Likely to Earn Its Place |
|---|---|---|---|
| **Hybrid RRF Search** | Corpus A: +0.0969 (PASS)<br>NFCorpus: +0.0137 (FAIL)<br>HotpotQA: +0.0250 (PASS) | [`reports/experiment_001.json`](reports/experiment_001.json)<br>[`reports/r06b_baseline.json`](reports/r06b_baseline.json) | High vocabulary mismatch between queries and corpus documents (e.g., lay queries searching technical literature). |
| **Cross-Encoder Reranking** | nDCG@10: +0.0737 (PASS)<br>Recall@10: +0.0059 (FAIL) | [`reports/experiment_002.json`](reports/experiment_002.json) | High candidate pool size where initial retrieval places relevant items in top-100 but misorders top-10. |
| **Per-Class Fusion Weights** | +0.0000 on Recall@10 | [`reports/experiment_002.json`](reports/experiment_002.json) | Distinct query classes with starkly different structural characteristics (e.g., pure keyword lookups vs complex semantic QA) and a highly accurate classifier ($>90\%$ accuracy). |
| **Self-Query Prefiltering** | Corpus A: -0.0461 (Harmful)<br>NFCorpus: N/A (No schema) | [`reports/experiment_003.json`](reports/experiment_003.json)<br>[`reports/r06c1_transformations.json`](reports/r06c1_transformations.json) | Corpora with 100% structured payload metadata coverage and strict multi-attribute query constraints where filter extraction precision approaches 1.0. |
| **HyDE (Hypothetical Embeddings)** | Corpus A: -0.0029<br>NFCorpus: -0.0026 | [`reports/experiment_005.json`](reports/experiment_005.json)<br>[`reports/r06c1_transformations.json`](reports/r06c1_transformations.json) | Abstract or conceptual queries where gold documents sit far from raw queries in vector space and distractor density is low. |
| **Query Decomposition** | Corpus A: -0.0100<br>NFCorpus: +0.0012 | [`reports/experiment_005.json`](reports/experiment_005.json)<br>[`reports/r06c1_transformations.json`](reports/r06c1_transformations.json), [`reports/r06d.json`](reports/r06d.json) | Complex, multi-part, multi-hop queries requiring explicit sub-question extraction, provided original query context is preserved during fusion. |
| **Multi-Representation Indexing** | NFCorpus: +0.0044 ($p=0.1592$)<br>Corpus A: Unresolvable (8.75%) | [`reports/r06c2_multirep.json`](reports/r06c2_multirep.json)<br>[`reports/r03_tiers.json`](reports/r03_tiers.json) | Long, heterogeneous documents where chunk-level summaries or extracted question representations provide distinct semantic entry points at $\ge 90\%$ index coverage. |
| **Cascade Router** | Cost saving: -0.51%<br>Max Oracle Saving: 36.06% | [`reports/r02f_oracle.json`](reports/r02f_oracle.json)<br>[`reports/r02b_gate.json`](reports/r02b_gate.json) | Systems where retrieval accounts for $>50\%$ of total query processing cost or where a high-precision escalation signal ($\text{ROC AUC} > 0.85$) is available. |
