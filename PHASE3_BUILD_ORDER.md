# PHASE 3 BUILD ORDER — Query Transformation

Hand this whole file to the coding agent. Written to be executed, not discussed.
Facts in §2 verified 2026-09-14.

---

## 0. Standing rules — these override any instinct to move fast

1. **Cross-check twice.** Before running a command or committing to a design, confirm it
   will not be wasted work. State the check you ran.
2. **Pre-empt LLM build errors.** §3 lists mistakes already made in this repo. Check your
   output against that list before returning it.
3. **First-try completion, correctness first.** No "we'll fix it next pass".
4. **Verify your own output.** Run the tests and the checker scripts. Read the result. Do
   not report success from an exit code. **Do not restate an acceptance criterion so that
   your result passes it** — if it fails, say it failed and report the gap.
5. **Token discipline.** Do not re-read files you already have. Do not print whole files
   back. Batch independent commands. One round trip per task.

---

## 1. Where the project stands

**Phases 1 and 2 complete.** 428 documents → 12,074 chunks → 22,422 representations in
`retriever_chunks_a5d51992`. Three-stage funnel working. 109 tests. Shipped config measured
against a fixed baseline:

```
Recall@10  0.3817 → 0.4793   +0.0976 [+0.0443, +0.1509]
nDCG@10    0.4344 → 0.5890   +0.1546 [+0.0742, +0.2389]
MRR        0.2607 → 0.3483   +0.0877 [+0.0378, +0.1371]
```

**Phase 3 scope:** the query transformation layer, routed by query class.

```yaml
query:
  transformation:
    enabled: true
    fixed: none
    mapping:
      exact_lookup: none        # literal strings - do not paraphrase them
      conceptual: hyde
      compound: decompose
      filterable: self_query
      multihop: none            # Phase 4 handles these with IRCoT
      aggregation: self_query
    vocab_overlap_threshold: 0.35
```

Nothing in that block is implemented. `vocab_overlap_threshold` is currently dead config.

**Two things Phase 3 must fix that Phase 2 left inert:**

- The structured pre-filter has returned `no_constraints` on **every query ever run**,
  because nothing produces `extracted_filters`. `self_query` is what makes stage 1 live.
- Six query classes are classified but only two behaviours differ between them (fusion
  weights). Transformation is what makes the classification earn its place.

---

## 2. Hard environment facts — do not rediscover these

- **Indexed payload fields:** `doc_class`, `contains_table`, `entities`, `date_refs`,
  `access_tags`, `kind`, `tier`. Nothing else may appear in a filter — `chunk_id` is not
  indexed and filtering on it is a linear scan over 22,422 points.
- **The LLM is `Qwen/Qwen2.5-0.5B-Instruct` on a GTX 1650, fp32.** It is small. Do not
  assume reliable structured JSON output from it — parse defensively and fall back to no
  transformation rather than to a malformed one.
- **`ingest_config_hash` must stay `a5d51992`.** It covers `ingest`,
  `models.dense_embedding`, and `models.llm` minus `device`/`batch_size`/`dtype`. Editing
  anything under `query`, `funnel` or `eval` does not move it. If your change moves it, you
  edited the wrong key and have forced a needless re-index.
- **`fuse_rrf` already accepts N named sources.** Decomposition does not need new fusion
  code — pass each sub-query's ranked list as its own source.
- **`tests/conftest.py` fails any test that writes under `data/`.** Use `tmp_path`.
- **No string literal over 200 characters outside `prompts.py`.** Adjacent literals are
  concatenated at compile time, so splitting across lines does not help. Shorten it.
- **All prompts live in `prompts.py`** and are loaded by `load_prompt_template`. There is a
  test enforcing this.
- **`ruff check --select F821,F811,E9 src tests scripts` must be clean.**

---

## 3. Failure catalogue — mistakes already made in this repo

Each of these shipped, reported success, and did nothing.

**Silent substitution**
1. Dense vectors were SHA-256 bytes; sparse vectors were process-unstable hash buckets.
2. The embedder fell back to a stub on load failure.
3. The reranker returned `[1.0] * n` on load failure.
4. A source missing from the RRF weights silently defaulted to `1.0` — double a normal 0.5.

**Swallowed errors**
5. An exception handler that caught the exception it had just raised.
6. A `NameError` swallowed so `parse_cache: {hits: 428, misses: 0}` was a lie.
7. A length-changing filter on a batch result, silently misaligning it with its inputs.

**Wrong unit of work**
8. RRF fused representations rather than chunks, so one chunk filled several top-k slots.
9. `candidate.text` carried the matched representation's text, so an LLM-written summary
   would have been cited as source evidence.

**Config that looks wired but is not**
10. `max_candidates` was read and copied into the trace but never applied to anything —
    a trace reporting a cap that was never enforced.
11. `gate_score_margin: 0.15` was compared against RRF scores whose maximum is 0.0164, so
    the gate could never close. "Gated" was silently "always".
12. `class_confidence` was a hardcoded constant presented as a computed score.

**Dishonest reporting**
13. `eval-verify` stamped `synthetic_verified` on `rows[:50]` with no human review.
14. 24 items carried "verification" timestamps 1.25 seconds apart.
15. A warm-up acceptance criterion was restated from "Run 1 within 20% of Run 3" to
    "Run 2 within 20% of Run 3" so the result would pass.
16. A config value was justified by a "6.5× latency cliff" that was an artifact of taking a
    median on a bimodal distribution.

**Tests that could not fail**
17. An expected value (`0.9`) coincidentally equalled an unrelated mock input (`0.9`).
18. No test loaded the real `config/default.yaml`, so "this value is wired up" was unproven.
19. A blanket `return None` mutation that failed every test in a file, proving nothing.

**The rule:** every new behaviour needs a test that fails when that behaviour is removed.
Prove it with a **targeted one-line mutation** — name the line you changed and the test that
caught it. A mutation that fails every test in the file is not evidence.

---

## 4. Task A — transformation interface and routing

**File:** `src/adaptive_retriever/query/transform.py`

```python
@dataclass
class TransformResult:
    strategy: str                 # "none" | "hyde" | "decompose" | "self_query"
    queries: List[str]            # >= 1; ["original"] for "none"
    extracted_filters: Dict[str, Any]
    trace: Dict[str, Any]         # what was done, why, elapsed_ms, cache hit/miss

def transform_query(fingerprint, cfg, llm=None) -> TransformResult: ...
```

**Rules:**

- Route by `query.transformation.mapping[query_class]`. An unmapped class raises, same as
  `class_weights` — and add a test deriving the required set from the `QueryClass` enum so
  the config cannot drift from the code.
- `enabled: false` returns `strategy="none"` with `trace["skipped"] = True`.
- `fixed: <strategy>` overrides the mapping for every class. This is the ablation switch
  Phase 6 needs — make it work now, and test it.
- **Every transformation must be cached** on `(query_text, strategy, model, prompt_version)`.
  An uncached HyDE call per query turns a 3-minute eval run into an hour. Reuse the existing
  LLM cache.
- Wire into `run_funnel` between the profiler and the pre-filter. `extracted_filters` feeds
  `build_prefilter`; `queries` feeds hybrid search.
- Record the transformation in the funnel trace as its own stage, with input/output counts
  and `elapsed_ms`, like every other stage.

**Tests:** routing per class from real config; unmapped class raises; `enabled: false`
short-circuits; `fixed` overrides mapping; cache hit on the second identical call.

---

## 5. Task B — `self_query` (do this before HyDE)

Extract structured filters from the query text into `extracted_filters`.

**Why first:** stage 1 has returned `no_constraints` on every query the system has ever run.
This is the task that makes a whole funnel stage do something, and it needs no generation if
done rule-first.

**Rules:**

- **Only the seven indexed fields may be produced.** Anything else is dropped into
  `trace["rejected_fields"]`, never into the filter.
- **A filter value that matches nothing is worse than no filter** — it silently collapses
  recall to zero. Before returning filters, count matching points; if zero, drop the filter,
  record `trace["dropped_reason"] = "matches_no_documents"`, and proceed unfiltered. Add a
  test for exactly this.
- Rule-based extraction first (explicit `doc_class` terms, date patterns, table mentions).
  Use the LLM only as a fallback, and parse its output defensively — Qwen-0.5B will produce
  malformed JSON. A parse failure means no filter, never a guessed one.
- Validate extracted values against what is actually in the index (`doc_class` has seven
  known values — reject anything else).

**Tests:** an unindexed field never reaches the filter; a value matching zero points is
dropped with a recorded reason; malformed LLM output yields no filter and no exception;
a valid `doc_class` mention produces the right condition.

---

## 6. Task C — `hyde`

Generate a hypothetical answer passage, embed **that** instead of the query.

**Rules:**

- Prompt goes in `prompts.py` with a version string. The version must be part of the cache key.
- Cap generation at `max_new_tokens: 128`. Longer output costs time and adds drift.
- If generation fails or returns empty, fall back to the original query and record
  `trace["fallback"] = "generation_failed"`. Never return an empty query string.
- **Wire `vocab_overlap_threshold: 0.35`, or delete the key.** Its intent is to apply HyDE
  only when the query shares little vocabulary with the corpus — high overlap means the
  literal query already matches. If you wire it, record the measured overlap in the trace.
  Do not leave it read-but-unused; see failure #10.
- Report the added latency per query. On this hardware a 128-token generation is on the
  order of 1 s and will dominate the funnel.

**Tests:** transformed query differs from the original (a transformation that returns the
input is dead code); generation failure falls back cleanly; cached second call issues no
generation; overlap above threshold skips HyDE if the threshold is wired.

---

## 7. Task D — `decompose`

Split a compound query into sub-queries, retrieve for each, fuse.

**Rules:**

- Cap at 3 sub-queries. Each additional sub-query is a full hybrid retrieval.
- **Reject degenerate decompositions.** If a sub-query is the original, or a near-duplicate
  of another sub-query, drop it and record why. A "decomposition" that returns the original
  three times triples cost for nothing — test for it explicitly.
- Fuse sub-query results with the existing `fuse_rrf`, one named source per sub-query, equal
  weights summing to 1.0. Do not write new fusion code.
- Collapse to chunks before fusion, as stage 2 already does.

**Tests:** degenerate sub-queries rejected; N sub-queries produce N named sources in the
fusion call; weights sum to 1.0; single surviving sub-query behaves as `none`.

---

## 8. Task E — Experiment 003

The measurement. Same eval set, same seed, same index, same baseline harness.

Report, as **paired differences against the Phase 2 shipped configuration** (transformation
off, `gate_score_margin: 0.30`):

1. Transformation on vs off, overall — Recall@10, nDCG@10, MRR, each with 95% CI.
2. The same, **broken down by query class**, with per-class n. Transformation is routed per
   class; an overall average can hide one strategy helping and another hurting.
3. `fixed: hyde`, `fixed: decompose`, `fixed: self_query`, `fixed: none` applied to all
   classes — the four single-strategy arms, so routing can be compared against each strategy
   used everywhere.
4. Added latency and LLM cost per query, per strategy, mean and p95.
5. For `self_query` specifically: how often a filter was produced, how often it was dropped
   for matching zero documents, and the pre-filter's output count distribution.

**State the multiplicity up front.** Count your comparisons before running, say how many
there are, and report whether anything survives correction. Do not select a winner from
overlapping intervals and call it confirmed.

**A negative result is a result.** If transformation does not help, report that plainly with
the intervals. Given the LLM is a 0.5B model, that is a genuinely likely outcome and it is
more useful than a manufactured win.

---

## 9. Acceptance gate

```powershell
.\.venv\Scripts\Activate.ps1
ruff check --select F821,F811,E9 src tests scripts
python -m pytest -q
python check_server_mode.py
python check_hybrid.py
python -m adaptive_retriever.cli funnel "compare the evaluation methods used across the studies" --trace
python scripts/run_experiment_003.py
```

Required:

- ruff clean; all tests pass; each new behaviour fails under its stated targeted mutation.
- `ingest_config_hash` still `a5d51992`.
- The funnel trace shows a **transformation stage** with a real strategy, and for a
  filterable query the pre-filter shows `decision: "applied"` with a non-empty
  `applied_constraints` — the first time that stage will have done anything.
- Experiment 003 table with per-class breakdowns and the multiplicity statement.

---

## 10. Reporting format

1. Files created or changed.
2. Tests added, and for each new behaviour the **targeted one-line mutation** you injected
   and which test caught it.
3. Acceptance gate output, trimmed to verdict lines.
4. Anything you could not verify, stated plainly as unverified.

Do not report success for a step you did not run. Do not restate a criterion to make a
result pass.
