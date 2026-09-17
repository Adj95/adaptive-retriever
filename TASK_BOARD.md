# Task Board — The Adaptive Retriever

Every remaining task, in order. One task per agent message. Read
`AGENT_OPERATING_MANUAL.md` first; it is the reviewer.

**How to use this file:** find the first task whose status is not `DONE`. Do only that task.
Produce its **Exit output** exactly. Do not start the next task until the exit criteria are met
and confirmed.

---

## Status

| Phase | State |
|---|---|
| 1 — ingestion & indexing | DONE |
| 2 — retrieval funnel, baseline, metrics | DONE |
| 3 — query transformation | DONE (none shipped) |
| 5 — post-retrieval & generation | **IN PROGRESS — T-01** |
| 6 — ablations | not started |
| 4 — escalation & IRCoT (moved last) | not started |

Published result, unchanged by later phases:

```
Recall@10  0.3817 -> 0.4793   +0.0976 [+0.0444, +0.1510]
nDCG@10    0.4344 -> 0.5888   +0.1545 [+0.0788, +0.2319]
MRR        0.2607 -> 0.3482   +0.0876 [+0.0418, +0.1363]
```

---

## T-01 — Fix compression candidate set  *(BLOCKER)*

**Doc:** PHASE5_BUILD_ORDER.md §4 · **Blocked by:** nothing

**Do:** `compress()` is receiving ~21,585 sentence spans for a single query. Ten candidates at
~15 sentences each is ~150. It is compressing the **pre-rerank pool of 100 candidates** instead
of the final top-k. Find the call site in `scripts/run_experiment_006.py`, move compression to
**after** rerank on the final candidates only, and revert every device and precision change
made while chasing this: `dtype: float32`, all models on CUDA.

**Check:**
- `len(candidates)` at the `compress()` call site is ≤ `funnel.rerank.output_size` (10)
- `len(all_spans)` is in the low hundreds, not tens of thousands
- `python check_preflight.py` still reports 22,422 points and hash `a5d51992`
- `config/default.yaml` has `models.llm.dtype: float32`

**Exit output:** the two printed counts, the preflight block, and a 2-item smoke test with the
projected wall-clock for 169 items × 4 arms. **Stop there.** Per-item cost should be under
2 seconds; if it is not, the bug is not fixed.

---

## T-02 — Experiment 006: post-retrieval arms

**Doc:** PHASE5_BUILD_ORDER.md §8 · **Blocked by:** T-01

**Do:** Four arms over the 169 non-rejected eval items, same seed, cache cleared, raw ledger
streamed per item:

1. generation alone
2. + reorder
3. + compression
4. + compression + reorder

Report per arm, using **`is_relevant()`** for every relevance decision:

- citation coverage (% answers with ≥ 1 attributed citation, n ≥ 8)
- unsupported-sentence rate
- groundedness: max-cosine distribution of unsupported sentences **with the shuffled null
  baseline computed the same way**
- refusal rate, and refusal correctness
- added latency per arm, mean and p95, stating cache-order effects

Paired bootstrap differences of arms 2, 3, 4 against arm 1. State the comparison count
(4 metrics × 3 arms = 12) and whether anything survives Bonferroni.

**Check:** arm 1 and arm 3 differ only by compression, which removes 1.46% of text at
`T=0.40` — expect near-zero difference and say so if that is what appears. Any non-zero
difference on an arm pair whose transformation is a no-op means the arms differ in something
unnamed.

**Exit output:** the results table, the multiplicity statement, and `reports/experiment_006.json`.

---

## T-03 — Close Phase 5

**Blocked by:** T-02

**Do:** Update `README.md` with a Phase 5 section, and add Phase 3 and Phase 5 entries to
`LESSONS.md`.

The Phase 5 README section must state, honestly:

- Post-hoc span attribution ships; model-emitted citations measured **0/169** zero-shot and
  **1.78%** few-shot on Qwen2.5-0.5B.
- Attribution covers **31.95%** of answers (canonical zero-shot corpus, 430 sentences);
  **78.98%** of sentences have no ≥8-gram grounding.
- That is **not** invention: the shuffled control gives 0.59% coverage at n = 8 against 34.91%
  real — a 59× separation. The generator paraphrases from retrieved context.
- Compression at `T=0.40` removes 1.46% of text; no threshold on this corpus removes meaningful
  text without destroying gold passages (`T=0.45`: 9% removed, 2.3% of gold lost; `T=0.50`:
  26% removed, 10.3% lost).
- The LLM extractor tier was **deleted**: it fired on 174 candidates of which 3 contained a gold
  passage — 98% noise.
- Confidence is computed and varies (159 distinct values) but discriminates weakly; report the
  **AUC**, not a mean difference against an invented threshold. No gating unless AUC justifies it.

The LESSONS entries worth writing:

- **Three of four "this technique is harmful" findings were implementation bugs.** HyDE −0.196
  → −0.003 once routing was fixed; decomposition −0.204 → −0.010 once the original query was
  retained. Rule: before publishing a large negative effect, check the implementation against
  the technique's definition.
- **A number without a comparison is not evidence.** 18.3% classifier accuracy (vs 20.1%
  majority), 0.74 cosine (vs 0.56 null), 79.88% 4-gram coverage (vs 12.43% shuffled). Every one
  of these looked like a finding until the control was run.
- **The wrong interpreter is invisible.** Anaconda's global python with CPU-only torch produced
  a 7× slowdown reported as a hardware bottleneck.

**Exit output:** the diff summary, and `git commit` + `git push`.

---

## T-04 — Phase 6 ablation harness

**Doc:** PHASE6_BUILD_ORDER.md §2 · **Blocked by:** T-03

**Exit output:** harness + tests + 2-item smoke test with projection. No full run yet.

---

## T-05 — Experiment 007: representation ablation

**Doc:** PHASE6_BUILD_ORDER.md §3 · **Blocked by:** T-04

**Do:** The headline ablation of the project. All arms use the **`kind` payload index** — no
re-indexing, `ingest_config_hash` must not move.

**Exit output:** the 8-arm table with paired differences against the shipped arm, multiplicity
stated, `reports/experiment_007.json`.

---

## T-06 — Close Phase 6

**Blocked by:** T-05

**Exit output:** README results section rewritten with the ablation table; commit and push.

---

## T-07 — Repo publish  *(do this before T-08; it is what the deadline needs)*

**Blocked by:** T-06

**Do:** Final README pass, verify every published number traces to a file in `reports/`, make
the GitHub repo public.

**Check:** clone the repo fresh into a temp directory, follow the Quickstart, and confirm it
runs. A README whose instructions fail is worse than none.

**Exit output:** the fresh-clone transcript, and the public repo URL.

---

## T-08 — Fix multi-hop eval generation

**Doc:** PHASE4_BUILD_ORDER.md §2 · **Blocked by:** T-07

**Do:** 28 multi-hop items were excluded as unanswerable — the generator paired each query with
one relevant passage and one sampled at random. Phase 4's IRCoT hop cannot be evaluated without
genuine multi-hop items.

**Exit output:** count of regenerated items, the verification that both passages are required,
and the updated provenance breakdown.

---

## T-09 — Escalation ladder

**Doc:** PHASE4_BUILD_ORDER.md §3 · **Blocked by:** T-08

**Read the warning in that doc first.** Two of the three configured rungs are measured-useless.

---

## T-10 — IRCoT hop + Experiment 008

**Doc:** PHASE4_BUILD_ORDER.md §4 · **Blocked by:** T-09

---

## T-11 — Close Phase 4 and the project

**Blocked by:** T-10

**Exit output:** final README, final LESSONS, resume bullet updated with the final numbers,
commit and push.

---

## If time runs short

The project is already resume-complete at **T-03**. T-07 (publish) is the highest-value
remaining item because it is what a recruiter actually opens.

**Priority order if the deadline compresses:** T-01 → T-02 → T-03 → T-07, then stop.
Phases 6 and 4 add interview depth, not resume content.
