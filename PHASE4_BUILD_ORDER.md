# PHASE 4 BUILD ORDER — Escalation and IRCoT

Read `AGENT_OPERATING_MANUAL.md` first. Tasks T-08 to T-11 on the task board.

**Phase 4 is numbered out of order deliberately.** It was moved last, and measurement since has
weakened the case for it further. Read §1 before building anything.

---

## 1. Read this before starting

The configured escalation ladder is:

```yaml
escalation:
  ladder: [enable_rerank, apply_transformation, ircot_hop]
```

**Two of the three rungs are measured-useless:**

| Rung | Status |
|---|---|
| `enable_rerank` | Reranking adds **no coverage** — `+0.0059 [−0.0473, +0.0651]` on Recall@10 between always and never. Escalating to it cannot recover a missed document. |
| `apply_transformation` | All three transformations measured neutral or harmful. `self_query` −0.0461, `hyde` −0.0029, `decompose` −0.0100 (query retained) / −0.0857 (replaced). |
| `ircot_hop` | Untested. Needs multi-hop eval items that do not currently exist. |

So an escalation ladder built as configured would escalate a failed query to two rungs that
have been shown not to help, and one that cannot be evaluated.

**Two honest options:**

- **(a) Build only the `ircot_hop` rung**, after fixing the eval set (T-08). Drop the other two
  rungs from the config with a recorded reason.
- **(b) Do not build Phase 4.** Record in the README that the escalation ladder was designed,
  then cancelled because measurement showed two of its three rungs contribute nothing, and the
  third could not be evaluated on an eval set whose multi-hop items were unusable.

**(b) is a legitimate outcome and takes an hour.** A cancelled subsystem with a measured reason
is a stronger entry in this project's story than a built one that does nothing. Decide
explicitly and record the decision — do not drift into building it.

If choosing (a), continue below.

---

## 2. Task T-08 — fix multi-hop eval generation

28 items were excluded as **provably unanswerable**: the generator paired each multi-hop query
with one relevant passage and one sampled at random. Retrieval could never return the random
passage.

**Build a generator that produces genuine multi-hop items:**

- Pick two chunks that share an entity (the `entities` payload index makes this cheap).
- Generate a question that requires a fact from **each** chunk — one alone must not suffice.
- **Verify** each item before accepting it: retrieve with only passage A's content as the query
  and confirm the answer cannot be derived; repeat for B. Record the verification in the item.
- Mark provenance `synthetic_multihop_verified` with the verification evidence. Do **not**
  reuse `synthetic_verified` — that flag has a history of being fabricated in this project.
- Keep the 28 rejected items with their rejection reasons. Never delete eval history.

**Exit output:** count generated, count passing verification, the updated provenance breakdown,
and three example items showing both passages and why each is necessary.

**Check:** `is_relevant()` must find both passages for a generated item. If it finds only one,
the item is single-hop and must be rejected.

---

## 3. Task T-09 — escalation ladder

Only if option (a) was chosen.

**File:** `src/adaptive_retriever/escalation/ladder.py`

```yaml
escalation:
  enabled: true
  confidence: {method: score_margin, threshold: 0.25}
  ladder: [ircot_hop]          # the other two rungs removed, with a recorded reason
  max_hops: 3
  cost_ceiling_usd: 0.05
  latency_ceiling_ms: 8000
```

**Rules:**

- **Escalation triggers on a confidence score.** That score's AUC was measured in Phase 5 and is
  weak. If AUC < 0.65, escalation fires close to randomly — say so in the trace and in the
  report, and do not present the ladder as adaptive.
- `cost_ceiling_usd` and `latency_ceiling_ms` must be **enforced**, not recorded. This repo has
  shipped four config keys that were read and never applied. Add a test that a query exceeding
  the ceiling stops escalating.
- Every escalation decision goes in the trace: rung, trigger value, threshold, outcome.

---

## 4. Task T-10 — IRCoT hop and Experiment 008

**IRCoT:** retrieve, generate a reasoning step, use it to retrieve again, repeat to `max_hops`.

**Rules:**

- Each hop is a full funnel pass. At ~1.3 s/query plus generation, 3 hops is ~15 s. Measure and
  report; do not assume.
- Deduplicate candidates across hops by `chunk_id` before fusion — the same collapse rule as
  stage 2.
- A hop that produces no new candidates terminates the ladder. Test this.
- Cache every generation by `(query, hop index, prior context, prompt_version)`.

**Experiment 008**, on the verified multi-hop subset only:

| Arm | |
|---|---|
| 1 | no escalation (shipped) |
| 2 | escalation with `ircot_hop`, max 1 hop |
| 3 | escalation with `ircot_hop`, max 3 hops |

Report Recall@10, nDCG@10, MRR with paired bootstrap differences against arm 1, plus latency
and generation cost per query. State the multiplicity (2 arms × 3 metrics = 6).

**Report the n.** The verified multi-hop subset will be small — likely 20–40 items. With n that
small, confidence intervals will be wide and almost nothing will reach significance. **Say this
before the results, not after.** An underpowered experiment honestly labelled is fine; one
presented as conclusive is not.

---

## 5. Task T-11 — close Phase 4 and the project

- README: final architecture, final results, the full list of what measurement removed.
- LESSONS: final entries.
- Resume bullet updated with the final numbers.
- Every published number traced to a file in `reports/`.

**The project's summary claim, which should survive all of this unchanged:**

> A hybrid multi-representation retrieval stack that beats a fixed dense baseline by +26%
> Recall@10 and +36% nDCG@10 (paired bootstrap 95% CI excludes zero, n = 169), in which five
> of six adaptive components were measured and removed.

---

## 6. Acceptance gate

```powershell
python check_preflight.py
ruff check --select F821,F811,E9 src tests scripts
python -m pytest -q
python scripts/run_experiment_008.py --smoke
python scripts/run_experiment_008.py
```

Required: preflight PASS; `ingest_config_hash` still `a5d51992`; cost and latency ceilings
demonstrated to actually stop escalation; the multi-hop subset size stated before any result.
