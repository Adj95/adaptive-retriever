# PHASE 5 BUILD ORDER — Grounded Generation

Hand this whole file to the coding agent. Written to be executed, not discussed.
Facts in §1 and §2 verified 2026-09-15.

**Note on numbering:** Phase 4 (escalation ladder, IRCoT) was moved last by design and is
now further devalued — see §1. Phase 5 follows Phase 3.

---

## 0. Standing rules

1. **Cross-check twice.** Before running a command or committing to a design, confirm it
   will not be wasted work. State the check you ran.
2. **Pre-empt known failures.** §3 lists mistakes already made in this repo. Check your
   output against that list before returning it.
3. **First-try completion, correctness first.**
4. **Verify your own output.** Read the result. Never report success from an exit code, and
   never restate a criterion so a result passes it.
5. **Token discipline.** Do not re-read files you have. Batch independent commands.

### Experiment hygiene — non-negotiable, all five added after real losses

6. **Verify every signature before calling it.** Two full runs were lost to
   `run_funnel(limit=...)` and `bootstrap_ci(n_resamples=...)`. Grep the definition.
7. **Smoke-test on 2 items end to end** — including aggregation, bootstrap and file writing —
   before any full-set run. Both losses above would have surfaced in ten seconds.
8. **Persist raw per-query results as they are produced**, to `reports/<experiment>_raw.jsonl`.
   Aggregate from that file. A statistics bug must never cost the compute again.
9. **Clear the transformation and generation caches at the start of any experiment** whose
   arms differ in what they cache. A stale cache silently froze one arm across two reports.
10. **Embed consistency assertions in the experiment script** and fail on violation. Every
    reported number must trace to a row written by that run.

---

## 1. Where the project stands

**Phases 1–3 complete.** 428 documents → 12,074 chunks → 22,422 representations in
`retriever_chunks_a5d51992`. 117 tests. Measured against a fixed dense-retrieval baseline:

```
Recall@10  0.3817 → 0.4793   +0.0976 [+0.0444, +0.1510]
nDCG@10    0.4344 → 0.5888   +0.1545 [+0.0788, +0.2319]
MRR        0.2607 → 0.3482   +0.0876 [+0.0418, +0.1363]
```

**What measurement has removed so far:**

| Component | Verdict |
|---|---|
| Per-class fusion weights | No effect. Fixed 0.5/0.5 vs routing: `+0.0000` Recall@10, p = 1.000. Removed. |
| Cross-encoder reranking (coverage) | No effect. Always vs never: `+0.0059 [-0.0473, +0.0651]`. Kept for ordering only. |
| Structured pre-filter (`self_query`) | Harmful. All five metrics negative; nDCG@10 `-0.0461 [-0.1045, -0.0036]`. Not shipped. |
| Query classifier | Unmeasurable — eval labels were assigned by generation rota. No longer routes anything. |

**Why Phase 4 stays last.** Its ladder is `[enable_rerank, apply_transformation, ircot_hop]`.
Two rungs are measured-useless, and `ircot_hop` depends on multi-hop eval items that were
excluded as unanswerable. Do not build it until those are fixed and a rung is shown to help.

**What Phase 5 adds.** The system currently retrieves and stops. There is no answer, no
citation, and no refusal. Phase 5 is what makes it a RAG system rather than a retriever.

---

## 2. Hard environment facts

- **The generator is `Qwen/Qwen2.5-0.5B-Instruct`, fp32, on a GTX 1650.** It is small. Expect
  short, sometimes wrong answers. The goal of this phase is **not** answer quality — it is a
  generation path whose faithfulness is *measurable*. Report what the model actually does.
- **Do not use a 0.5B model as its own judge.** `eval.judge` is configured, but a 0.5B model
  scoring a 0.5B model's answers produces noise dressed as a metric. See §8 — Phase 5 uses
  deterministic checks. Leave `eval.judge` unwired and say why in the report.
- **`RetrievalCandidate.text` is the chunk's RAW text**, hydrated by point-id lookup. Never
  cite a summary or question representation as source evidence — that bug already shipped once.
- **All prompts live in `prompts.py`** with a version string, loaded by `load_prompt_template`.
  A test enforces this. **No string literal over 200 characters outside `prompts.py`** —
  adjacent literals concatenate at compile time, so splitting lines does not help.
- **`ingest_config_hash` must stay `a5d51992`.** Editing anything under `postretrieval`,
  `generation`, `riskgate` or `eval` does not move it.
- **`tests/conftest.py` fails any test that writes under `data/`.** Use `tmp_path`.
- **`ruff check --select F821,F811,E9 src tests scripts` must be clean.**

---

## 3. Failure catalogue

Every one of these shipped, reported success, and did nothing. Check your work against them.

**Silent substitution** — hash bytes as embeddings; a stub embedder on load failure; a
reranker returning `[1.0] * n`; an unweighted RRF source defaulting to 1.0; **a server-mode
config silently opening a local store and running a 30-minute experiment against the wrong
backend.**

**Swallowed errors** — a handler catching the exception it raised; a `NameError` making
`parse_cache: {hits: 428, misses: 0}` a lie; a length-changing filter misaligning a batch.

**Wrong unit** — RRF ranking representations instead of chunks; a summary's text returned as
source evidence.

**Config that looks wired but isn't** — `max_candidates` reported in a trace while capping
nothing; `class_confidence` a constant presented as a score; `corpus_vocab_overlap`
hardcoded to 0.5; `gate_score_margin: 0.15` compared against RRF scores capped at 0.0164, so
the gate could never close.

**Dishonest evaluation data** — verification stamped on `rows[:50]` with no human review;
multi-hop items paired with random passages; class labels assigned by a generation rota.

**Statistics that mislead** — a median flipping modes on a bimodal distribution read as a
6.5× latency cliff; overlapping marginal CIs used in place of paired ones; a winner selected
from 18 comparisons and called confirmed.

**Tests that cannot fail** — an expected value coinciding with an unrelated mock input; no
test loading the real config; a blanket `return None` mutation failing every test in a file.

**Reporting failures** — an acceptance criterion restated so the result passed; a crashed job
reported as running; a report asserting `Filters Produced: 0` while its own prefilter mean
proved 24 filters were applied; one arm's numbers carried over from a previous run.

**The rule:** every new behaviour gets a test proven to fail when that behaviour is removed,
demonstrated by a **targeted one-line mutation** naming the line changed and the single test
that caught it.

---

## 4. Task A — post-retrieval compression

**File:** `src/adaptive_retriever/postretrieval/compress.py`

```yaml
postretrieval:
  compression:
    mode: cascade              # cascade | embeddings | none
    embeddings_threshold: 0.35
    llm_extractor_if_below: 0.5
```

```python
def compress(query: str, candidates: List[RetrievalCandidate], cfg) -> Tuple[List[RetrievalCandidate], Dict]:
    """Drop or shorten candidate text that does not bear on the query."""
```

**Rules:**

- Cascade: sentence-level embedding similarity first; fall to the LLM extractor only when the
  best sentence similarity is below `llm_extractor_if_below`. Record which tier fired per
  candidate.
- **Compression must never invent text.** Every surviving span must be a verbatim substring
  of the original chunk. Assert it, and add a test that a compressor returning paraphrased
  text fails.
- Track and report the compression ratio (characters out / characters in) per candidate.
- `mode: none` is a pass-through and must be exercised by a test.

**The honest risk:** compression that drops the sentence containing the answer costs recall
silently. Measure it — §8 compares answer quality with compression on and off.

---

## 5. Task B — context reordering

**File:** `src/adaptive_retriever/postretrieval/reorder.py`

`postretrieval.reorder.enabled: true`. Language models attend less to the middle of a long
context, so place the highest-scoring candidates at the start and end and the weakest in the
middle.

**Rules:**

- Reordering changes only position. Assert that the set of candidate ids before and after is
  identical — a reorder that drops a candidate is a bug, not a reorder.
- Record the mapping from pre-order rank to post-order position in the trace.
- Deterministic for a fixed input.

**Tests:** set identity before/after; the top candidate lands first; `enabled: false` is
order-preserving; mutation — make reorder drop the last candidate, the set-identity test must
catch it.

---

## 6. Task C — grounded generation with citations

**File:** `src/adaptive_retriever/generation/generate.py`

```yaml
generation:
  enabled: true
  cite_chunks: true
  refuse_if_confidence_below: 0.15
```

```python
@dataclass
class GeneratedAnswer:
    text: str
    citations: List[str]          # chunk_ids actually cited
    refused: bool
    refusal_reason: Optional[str]
    trace: Dict[str, Any]
```

**Rules:**

- The prompt goes in `prompts.py` with a version string, and that version is part of the
  generation cache key.
- **Citations must be resolvable.** Every chunk_id in `citations` must appear in the
  candidates passed in. A citation to a chunk that was not in context is a hallucinated
  citation — raise, do not filter it out quietly.
- Parse citations defensively. Qwen-0.5B will produce malformed markers; a parse failure
  means zero citations and a recorded reason, never a guessed chunk_id.
- Cache on `(query, ordered chunk_ids, prompt_version, model)`.
- Generation failure returns a refusal with `refusal_reason`, never an empty answer string.

---

## 7. Task D — the risk gate

```yaml
generation:
  refuse_if_confidence_below: 0.15
riskgate:
  enabled: true
  flag_on:
    low_confidence_below: 0.3
    contradictory_sources: true
    escalation_exhausted: true
```

**Rules:**

- **`confidence` must be a computed number, not a constant.** This codebase has already
  shipped `class_confidence` as a hardcoded 0.878 presented as a score. Derive it from the
  retrieval score margin (`escalation.confidence.method: score_margin`), document the formula
  in the docstring, and report its distribution across the eval set split by whether the
  answer was correct. **If confidence does not separate right from wrong answers, say so and
  do not gate on it.**
- `contradictory_sources`: detect when two cited chunks disagree. Start with a cheap
  deterministic signal (contradictory numbers or dates for the same entity). Do not use the
  0.5B model as a contradiction detector without measuring it.
- A refusal is a first-class outcome, not an error. Report the refusal rate, and how often a
  refusal was correct — i.e. the gold passage genuinely was not retrieved.

---

## 8. Task E — Experiment 006, answer faithfulness

This is the phase's real deliverable. **Do not use an LLM judge.** Four deterministic metrics,
all computable without a second model:

1. **Citation precision** — of the chunks cited, how many contain text supporting the answer?
   Measure with verbatim n-gram overlap (n ≥ 8) between answer spans and cited chunk text.
2. **Citation recall of the gold passage** — when the gold passage was retrieved, how often
   is it among the cited chunks?
3. **Answer groundedness** — fraction of the answer's content words that appear in the cited
   chunks. Low groundedness with high fluency is the signature of a model answering from its
   own weights rather than the context.
4. **Refusal correctness** — refusal rate, and the share of refusals where the gold passage
   was genuinely absent from the retrieved set.

**Arms**, all on the same 169 items, same seed, same cache-cleared protocol, paired bootstrap
differences against arm 1:

1. retrieval → generation (no compression, no reorder)
2. + reorder
3. + compression
4. + compression + reorder

State the comparison count up front (4 metrics × 3 arms = 12) and whether anything survives
correction.

**Also report**: added latency and generation cost per query, mean and p95; and the
groundedness distribution, since a 0.5B model producing ungrounded answers is the single most
likely outcome and is worth reporting as a finding rather than hiding.

---

## 9. Acceptance gate

```powershell
.\.venv\Scripts\Activate.ps1
ruff check --select F821,F811,E9 src tests scripts
python -m pytest -q
python check_server_mode.py
python check_hybrid.py
python -m adaptive_retriever.cli answer "what methods were used to evaluate results" --trace
python scripts/run_experiment_006.py --smoke
python scripts/run_experiment_006.py
```

Required:

- ruff clean; all tests pass; each new behaviour fails under its stated targeted mutation.
- `ingest_config_hash` still `a5d51992`.
- The `answer` command prints an answer, its citations, and for each citation the chunk_id
  plus the supporting span — so a reader can verify the citation by eye.
- Experiment 006 table with the multiplicity statement, produced by a run whose raw ledger
  exists and whose consistency assertions passed.

---

## 10. Reporting format

1. Files created or changed.
2. Tests added, and for each new behaviour the **targeted one-line mutation** injected and the
   single test that caught it.
3. Acceptance gate output, trimmed to verdict lines.
4. Anything you could not verify, stated plainly as unverified.

Do not report success for a step you did not run. Do not restate a criterion to make a result
pass. Read the exit code of every task you launch before describing its status.
