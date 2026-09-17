# Agent Operating Manual — The Adaptive Retriever

This file replaces the human reviewer. Read it in full before any task, and check every
report you produce against §3 before sending it.

---

## 1. Standing rules

1. **Cross-check twice.** Before running a command or committing to a design, confirm it will
   not be wasted work. State the check you ran.
2. **Pre-empt known failures.** §4 lists mistakes already made in this repo. Check your output
   against that list.
3. **First-try completion, correctness first.** No "we'll fix it next pass".
4. **Verify your own output.** Read the result. Never report success from an exit code, and
   never restate a criterion so a failing result passes it.
5. **Token discipline.** Do not re-read files you have. Batch independent commands.
6. **Verify every function signature before calling it.** Two full runs were lost to
   `run_funnel(limit=...)` and `bootstrap_ci(n_resamples=...)`. Grep the definition.
7. **Smoke-test on 2 items end to end** — including aggregation, bootstrap and file writing —
   then **report the measured per-item cost and projected total wall-clock and STOP.** Do not
   launch a full run until confirmed. If the projection exceeds 45 minutes, say so explicitly.
8. **Persist raw per-query results as they are produced** to `reports/<experiment>_raw.jsonl`.
   Aggregate from that file. A statistics bug must never cost the compute.
9. **Clear caches at the start of any experiment** whose arms differ in what they cache.
10. **Embed consistency assertions in the experiment script** and fail on violation.
11. **Never change device or precision without a measurement showing it is necessary.**
    Four consecutive device changes once cost a full day. `check_gpu_now.py` measured the
    working configuration: cross-encoder + Qwen2.5-0.5B fp32, both on CUDA, peak 2,356 MB of
    4,295 MB. It fits.
12. **Read the exit code of every task you launch before describing its status.** A crashed
    job reported as running has happened twice.

---

## 2. Hard environment facts

- Interpreter must be `.venv\Scripts\python.exe`. Anaconda's global python has a CPU-only
  torch and once caused a 7× slowdown that was misreported as a hardware finding.
- `models.llm.dtype: float32`. **fp16 is 3.5× slower on this GTX 1650** — no tensor cores.
  This is measured, not assumed. Do not "optimise" it to fp16.
- All models on CUDA. Moving them to CPU exhausts system RAM.
- `ingest_config_hash` must stay **`a5d51992`**. It covers `ingest`,
  `models.dense_embedding`, and `models.llm` minus `device`/`batch_size`/`dtype`. Changes
  under `funnel`, `query`, `postretrieval`, `generation`, `riskgate` or `eval` do not move it.
- Collection `retriever_chunks_a5d51992`, **22,422 points**, 7 payload indexes:
  `doc_class`, `contains_table`, `entities`, `date_refs`, `access_tags`, `kind`, `tier`.
  `chunk_id` is **not** indexed — filtering on it is a linear scan.
- Representation ids: `{chunk_id}_raw`, `_summary`, `_keywords`, `_question_{i}`.
  RAW text is byte-equal to the chunk text.
- `tests/conftest.py` fails any test that writes under `data/`. Use `tmp_path`.
- No string literal over 200 characters outside `prompts.py`. Adjacent literals concatenate
  at compile time, so splitting lines does not help.
- `ruff check --select F821,F811,E9 src tests scripts` must be clean.
- Relevance is decided by **`is_relevant()` in `metrics.py`** — token containment ≥ 0.60 AND a
  contiguous 8-gram verbatim match. Use it everywhere. Ad-hoc substring matching has produced
  three different "gold present" counts (85, 90, 101) across three reports.

---

## 3. Self-review checklist — run this on your own report before sending

Four reports in this project contained a summary contradicted by their own numbers. Check:

1. **Does any sentence in your summary contradict any number in your tables?**
   Past failures: `Filters Produced: 0` beside a prefilter mean below its median;
   `Queries with 0 candidates: 0` beside an unchanged mean.
2. **Did any number change from a previous report?** If yes, state the cause and which value
   is canonical. Past: `31.95% → 34.91%`, `78.98% → 88.89%`, `85 → 101 → 90`.
3. **Does every comparative claim have a control?** A similarity of 0.74 means nothing without
   a shuffled baseline. An accuracy of 18.3% means nothing without the majority-class rate.
4. **Is any threshold compared against a score whose real range makes it unreachable?**
   `gate_score_margin: 0.15` against RRF scores capped at 0.0164 could never fire.
   `llm_extractor_if_below: 0.50` against a minimum of 0.5403 never fired.
5. **Is any "computed" value actually constant?** Print the distinct-value count. Two subsets
   reporting identical statistics to four decimals means the value does not vary.
6. **Was a conclusion measured under configuration A applied to configuration B?** Gold
   survival measured at `T=0.35` does not justify shipping `T=0.50`.
7. **Is every mutation in production code, and does it break exactly one test?** Mutating a
   test proves nothing. A blanket `return None` that breaks every test proves nothing.
8. **Multiplicity:** how many comparisons did you run? Does anything survive Bonferroni?
   State the count before the results, not after.
9. **Paired differences, not overlapping marginal CIs.** Two arms' individual intervals
   overlapping is not a test.
10. **Does any latency number include cold start, cache state or arm ordering?** A shared
    cache makes the first arm pay for the rest. Say so.
11. **Is the interpreter, device and backend verified in the run that produced the numbers?**
    Print `sys.executable`, `torch.version.cuda`, and the resolved device at the point of use.
12. **Do the parts sum to the total?** Past: retrieval claimed at 17.5 min against 5.4 min of
    per-arm data; components summing to 118 min against a 66 min total.

---

## 4. Failure catalogue

Every one of these shipped, reported success, and did nothing.

**Silent substitution** — SHA-256 bytes as embeddings; a stub embedder on load failure; a
reranker returning `[1.0] * n`; an unweighted RRF source defaulting to 1.0; a server-mode
config silently opening a local store and running a 30-minute experiment against it.

**Swallowed errors** — a handler catching the exception it raised; a `NameError` making
`parse_cache: {hits: 428, misses: 0}` a lie; a length-changing filter misaligning a batch.

**Wrong unit** — RRF ranking representations instead of chunks; a summary's text returned as
source evidence; compression running on 100 pre-rerank candidates instead of the final 10.

**Config that looks wired but isn't** — `max_candidates` reported in a trace while capping
nothing; `class_confidence` a constant presented as a score; `corpus_vocab_overlap` hardcoded
to 0.5; a gate threshold that could never fire.

**Dishonest evaluation data** — verification stamped on `rows[:50]`; multi-hop items paired
with random passages; class labels assigned by a generation rota.

**Statistics that mislead** — a median flipping modes on a bimodal distribution read as a 6.5×
cliff; overlapping marginal CIs used in place of paired ones; a winner selected from 18
comparisons and called confirmed; a cosine of 0.74 called high with no null baseline.

**Implementation bugs reported as findings** — HyDE measured at −0.196 because the hypothetical
document went to the dense encoder, BM25 **and** the reranker; decomposition at −0.204 because
`queries[0]` was used and the rest discarded. Corrected, both are neutral.

**Reporting failures** — a criterion restated so the result passed; a crashed job reported as
running; a report asserting zero filters while its own mean proved otherwise; one arm's numbers
carried over from a previous run; an implementation changed between reports and presented as
"what the code does".

---

## 5. Report format

1. **Preflight block**, pasted verbatim from `python check_preflight.py`.
2. Files created or changed.
3. Tests added, and for each new behaviour the **targeted one-line production mutation** you
   injected and the **single** test that caught it.
4. Results, with the multiplicity count stated before the numbers.
5. Acceptance gate output, trimmed to verdict lines.
6. **Anything you could not verify, stated plainly as unverified.**

Then run §3 against what you just wrote, and fix it before sending.
