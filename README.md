# The Adaptive Retriever

> Every retrieval technique has a cost and a condition under which it pays. Most systems guess the condition. This one measures it, and defaults to cheap.

---

## 1. Thesis & Overview

Existing retrieval-augmented generation (RAG) architectures commit to a fixed combination of chunking strategy, index representations, search transformations, and reranking pipelines regardless of input query or document characteristics. **The Adaptive Retriever** dynamically fingerprints documents at ingest and queries at runtime, routing work to the cheapest strategy that satisfies a confidence gate. Crucially, every adaptive routing claim is empirically evaluated against a rigid, unrouted baseline (`baseline_fixed.yaml`).

---

## 2. Architecture & Pipeline Flows

```
INGEST FLOW:
[Document] -> [Profiler (Deterministic)] -> [Parser Router] -> [Chunk Router] 
           -> [Representation Tiering (RAW / SUMMARY / QUESTION / KEYWORDS)] 
           -> [Dual Persistence: SQLite registry + Qdrant Vector Store (dense + sparse)]

QUERY FLOW:
[Query] -> [Profiler (Deterministic)] -> [Transformation Router (none | hyde | step_back | decompose | self_query)]
        -> [3-Stage Funnel: Prefilter -> Hybrid Search (RRF) -> Gated Rerank]
        -> [Confidence Gate (score_margin / judge)]
        -> Pass? -> [Contextual Compression + Long-Context Reorder] -> [Cited Generation]
        -> Fail? -> [Escalation Ladder: Force Rerank -> Force Transform -> IRCoT Multi-Hop]
```

---

## 3. Headline Results (Test Split)

*To be populated after Phase 7 ablation run.*

| Strategy Arm | Recall@1 | Recall@5 | Recall@20 | nDCG@10 | MRR | p50 Latency (ms) | p95 Latency (ms) | Mean USD / Query |
|---|---|---|---|---|---|---|---|---|
| `baseline_fixed` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `full_adaptive` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

*(Note: All metrics reported with 95% bootstrapped confidence intervals [lower, upper]. All latency figures measured using `local_embedded` store mode).*

---

## 4. Funnel Cardinality & System Metrics

*To be populated after Phase 7 ablation run.*

- **Stage 1 (Prefilter)**: Mean candidates retained = *TBD*
- **Stage 2 (Hybrid RRF)**: Dense limit = 100, Sparse limit = 100, Fused candidates = *TBD*
- **Stage 3 (Gated Rerank)**: Input candidates = 100, Output candidates = 5 (Gated skip rate = *TBD*)

---

## 5. Cost & Latency Profile

- **Average Cost per Query**: *TBD USD*
- **p95 Latency (embedded client — not representative of server performance)**: *TBD ms*

---

## 6. Limitations & Negative Results

1. **Evaluation Set Ground Truth**: Ground truth items are partially generated using LLM prompts over canonical document parses with paragraph-scale sampling ($\ge 15$ tokens). Human verified items size: *TBD* (`provenance="synthetic_verified"`).
2. **Confidence Proxy**: The default `score_margin` confidence gate ($p[0] - p[1]$ score separation) is an uncalibrated proxy for answerability.
3. **Corpus Transferability**: Results are evaluated on a curated heterogeneous mix of 400–800 public documents across 6 domain classes; metrics may not transfer directly to specialized domain corpora.
4. **Vector Store Embedded Mode**: Qdrant execution uses local embedded persistent storage (`qdrant-client` path mode). In local mode, payload indexes have no effect (`UserWarning: Payload indexes have no effect in the local Qdrant`), confirming Addendum v1.2 §B3. Filtering functions correctly but runs unindexed, so funnel stage-1 prefiltering timings and latency figures reflect local Python embedded execution and will not be representative of a production Qdrant deployment.
5. **Environment Configuration**: API calls to SEC EDGAR require setting the environment variable `SEC_USER_AGENT="YourName email@domain.com"`.
6. **Failed Adaptive Arms**: *To be populated from the "What did not work" section of `reports/ablation.md`.*

---
