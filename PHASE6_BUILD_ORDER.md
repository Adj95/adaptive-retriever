# PHASE 6 BUILD ORDER — Ablations

Read `AGENT_OPERATING_MANUAL.md` first. Tasks T-04 to T-06 on the task board.

---

## 1. Why this phase is short

Most of what a Phase 6 ablation suite would test has already been measured:

| Component | Measured in | Result |
|---|---|---|
| Per-class fusion weights | Experiment 004 | No effect (p = 1.000) — removed |
| Rerank gate threshold | Experiment 002 | Flat for coverage; sweep exploratory |
| Query transformations | Experiments 003, 005 | None help; two neutral, one harmful |
| Post-retrieval compression / reorder | Experiment 006 | (pending T-02) |

**One thing has never been ablated, and it is the one that matters.** The project's entire gain
over baseline — `Recall@10 0.3817 → 0.4793` — comes from hybrid multi-representation retrieval.
Nobody has measured **which representations** contribute, or whether the hybrid is doing the
work rather than the representations.

That is Phase 6.

---

## 2. Task T-04 — the ablation harness

**File:** `scripts/run_experiment_007.py`

Every arm is a **payload filter over the existing index**. No re-indexing. `kind` is already a
payload index, so filtering is cheap and `ingest_config_hash` must stay `a5d51992`.

**Rules:**

- Arms are expressed as a `kind` filter plus a fusion-source selection. Nothing else changes.
- Reuse `run_funnel` — do not fork the retrieval path for the experiment.
- Raw ledger streamed per item to `reports/experiment_007_raw.jsonl`.
- Consistency assertions: the shipped arm must reproduce `Recall@10 = 0.4793` to within the
  documented ±2-query cross-encoder non-determinism. If it does not, stop.
- Relevance decided by `is_relevant()` only.

**Exit:** harness, tests with targeted production mutations, and a 2-item smoke test with the
projected wall-clock. **Stop before the full run.**

---

## 3. Task T-05 — Experiment 007

**Eight arms**, all on the same 169 items, same seed, cache cleared.

**Representation ablation** — which indexed views earn their generation cost:

| Arm | `kind` filter | Question it answers |
|---|---|---|
| 1 | `raw` only | Does multi-representation indexing help at all? |
| 2 | `raw` + `summary` | Do LLM summaries help? |
| 3 | `raw` + `question` | Do generated questions help? |
| 4 | `raw` + `keywords` | Do keywords help? |
| 5 | all four (**shipped**) | The reference arm |

**Fusion ablation** — is the hybrid doing the work:

| Arm | Sources | Question |
|---|---|---|
| 6 | dense only, all kinds | Is BM25 contributing? |
| 7 | sparse only, all kinds | Is the embedding contributing? |
| 8 | dense + sparse, `raw` only | Hybrid without multi-representation |

**Report** per arm: Recall@5, Recall@10, nDCG@10, MRR, and retrieval latency p50/p95. Paired
bootstrap differences against **arm 5** with 95% CIs.

**Multiplicity:** 7 arms × 4 quality metrics = **28 comparisons**. State this before the table,
and state which survive Bonferroni at α = 0.0018.

**What to watch for:**

- **Arm 1 vs arm 5** is the headline. If the difference is small, your 22,422-point index buys
  little over the 12,074 raw chunks, and the generation cost of summaries and questions was not
  repaid. That is a legitimate and publishable finding — the README currently attributes the
  gain to multi-representation indexing, and this is the test of that claim.
- **Arm 8 vs arm 1** isolates hybrid fusion from representations. Together with arm 1 vs arm 5,
  these two comparisons decompose the entire gain.
- Arms 6 and 7 must not beat arm 5 by much. If sparse-only wins, the corpus is lexical and the
  dense half is decorative — Experiment 004 already hinted at this (0.3/0.7 was numerically best).

**Do not tune anything on these results.** Report them. Tuning on the eval set contaminates
every number already published.

**Exit:** the 8-arm table, paired differences, multiplicity statement, `reports/experiment_007.json`.

---

## 4. Task T-06 — close Phase 6

Rewrite the README results section around the decomposition:

> The gain over a fixed dense baseline decomposes as follows: hybrid fusion contributes X,
> multi-representation indexing contributes Y, and reranking contributes ordering but not
> coverage.

If arm 1 ≈ arm 5, say so plainly and add to `LESSONS.md`: **generating 10,449 LLM
representations for a 428-document corpus did not repay its cost.** That is a more useful
finding than a positive one, and it is the kind of thing nobody reports about their own system.

**Exit:** README updated, LESSONS updated, commit and push.

---

## 5. Acceptance gate

```powershell
python check_preflight.py
ruff check --select F821,F811,E9 src tests scripts
python -m pytest -q
python scripts/run_experiment_007.py --smoke
python scripts/run_experiment_007.py
```

Required: preflight PASS with hash `a5d51992` and 22,422 points; arm 5 reproduces
`Recall@10 = 0.4793`; all tests green with targeted mutations stated.
