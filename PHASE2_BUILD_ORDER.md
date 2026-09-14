# PHASE 2 BUILD ORDER — The Adaptive Retriever

Hand this whole file to the coding agent. It is written to be executed, not
discussed. Every fact in §2 was verified on 2026-09-14, not assumed.

---

## 0. Standing rules — these override any instinct to move fast

1. **Cross-check twice.** Before running a command or committing to a design,
   confirm it will not be wasted work. State the check you ran.
2. **Pre-empt LLM build errors.** §3 lists the exact mistakes that have already
   been made in this repo. Check your output against that list before
   returning it.
3. **First-try completion, but correctness first.** No "we'll fix it in the next
   pass". A wrong-but-running implementation costs more than a slow one.
4. **Verify your own output.** Run the tests. Run the checker scripts. Read the
   result. Do not report success from the fact that a command exited 0.
5. **Token discipline.** Do not re-read files you already have. Do not print
   whole files back. Do not restate the plan. Batch independent commands into
   one call. One round trip per task, not five.

---

## 1. Where the project stands

**Phase 1: complete and verified.**

| fact | value |
|---|---|
| corpus | 428 documents, real, no synthetic text |
| chunks | 12,074 |
| indexed points | 22,422 (one per representation) |
| collection | `retriever_chunks_a5d51992` |
| Qdrant | server mode, `http://localhost:6333`, Docker |
| dense model | `BAAI/bge-small-en-v1.5`, 384-dim, cosine |
| sparse model | `Qdrant/bm25` via fastembed |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct`, cuda, **float32** |
| generations | 10,449, all cached |
| eval set | `data/eval/eval_set.jsonl`; 28 multihop items excluded as `synthetic_rejected` |
| tests | 81 passing |

**Phase 2: started, roughly one third done.**

Done:
- `funnel/fusion.py` — `fuse_rrf` and `collapse_to_best`, pure, 22 tests,
  mutation-verified.
- `VectorStore.query_hybrid` — dense + sparse, weighted RRF, representations
  collapsed to chunks before fusion, raw chunk text hydrated by point-id lookup.
- `funnel.hybrid.class_weights` complete for all six `QueryClass` members, each
  pair validated to sum to 1.0.

Not done — this is Phase 2's remaining scope:
- **A.** Structured pre-filter (stage 1 of the funnel).
- **B.** Gated rerank (stage 3).
- **C.** Funnel orchestrator that runs all three stages and emits a trace.
- **D.** Fixed baseline twin.
- **E.** Metrics harness with bootstrapped confidence intervals.

---

## 2. Hard environment facts — do not rediscover these

- **Payload indexes that exist:** `doc_class`, `contains_table`, `entities`,
  `date_refs`, `access_tags`, `kind`, `tier`. **`chunk_id` is NOT indexed.**
  Filtering on `chunk_id` is a linear scan over 22,422 points. Retrieve by point
  id instead — `stable_point_id(rep_id)` is deterministic.
- **Representation ids:** `f"{chunk_id}_raw"`, `f"{chunk_id}_summary"`,
  `f"{chunk_id}_keywords"`, `f"{chunk_id}_question_{i}"`. RAW text is byte-equal
  to the chunk text; this is an enforced invariant.
- **`ingest_config_hash` must stay `a5d51992`.** It covers `ingest`,
  `models.dense_embedding` and `models.llm` minus `device`/`batch_size`/`dtype`.
  Editing anything under `funnel`, `query`, `eval` or `vectorstore` does not
  move it. If your change moves it, you have forced a 10-minute re-index and you
  have almost certainly edited the wrong key.
- **GPU is a GTX 1650 (compute 7.5, no tensor cores).** fp16 is ~3.5× *slower*
  than fp32 here. Never set `dtype: float16`.
- **`tests/conftest.py` fails any test that opens a path under `data/` for
  writing.** Use `tmp_path`.
- **No string literal over 200 characters outside `prompts.py`.** Adjacent
  string literals are concatenated at compile time, so splitting a long message
  across lines does **not** satisfy this. Shorten it.
- **`ruff check --select F821,E9 src tests` must be clean** before you report
  done. This has caught 39 undefined names and a syntax error in one pass before.
- Reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

## 3. Failure catalogue — the mistakes already made in this repo

Read this before writing code. Each one shipped, reported success, and did
nothing.

**Silent substitution**
1. Dense vectors were SHA-256 hash bytes; sparse vectors were
   `hash(word) % 10000` buckets that were not even stable across processes.
   Measured cosine on a paraphrase pair: 0.0044.
2. The embedder fell back to a stub when the model failed to load.
3. The reranker returned `[1.0] * n` on load failure.
4. Hybrid query silently fell back to dense-only.
5. A source missing from the RRF weights silently defaulted to `1.0` — double a
   normal `0.5` source.

**Errors that were swallowed**
6. An exception handler that caught the exception it had just raised.
7. A `NameError` swallowed so that `parse_cache: {hits: 428, misses: 0}` was a
   lie.
8. A missing `continue` after a parse failure, so a failed parse fed downstream.
9. `[r for r in responses if r is not None]` changed the length of a batch
   result, silently misaligning it with its inputs.

**Wrong unit of work**
10. RRF fused *representations* rather than *chunks*, so one chunk could occupy
    several slots in the top-k and Recall was measured against a duplicated list.
11. `candidate.text` carried the matched representation's text, so an
    LLM-written summary would have been quoted and cited as source evidence.

**Dishonest evaluation**
12. `eval-verify` stamped `provenance: synthetic_verified` on `rows[:50]`
    without showing a single item to a human.
13. Multihop eval items paired one relevant passage with a randomly sampled one,
    making them unanswerable and silently depressing Recall.

**Tests that could not fail**
14. A test asserted an expected value (`0.9`) that coincidentally equalled an
    unrelated mock input (`0.9`), so it passed whichever the code used.
15. No test loaded the real `config/default.yaml`, so "this config value is
    wired up" was never actually proven.
16. A test encoded a wrong belief as an assertion (`vectorstore.collection` must
    change `ingest_config_hash`) and every assertion in it would have passed if
    the function returned a constant.

**The rule that falls out of this:** every new behaviour needs at least one test
that fails when the behaviour is removed. Prove it by removing the behaviour and
watching the test fail. Say in your report which mutation you tried.

---

## 4. Task A — structured pre-filter (funnel stage 1)

**File:** `src/adaptive_retriever/funnel/prefilter.py`

**Config already present:**
```yaml
funnel:
  prefilter:
    enabled: true
    max_candidates: 100000
```

**Build:**
```python
def build_prefilter(
    profile: QueryProfile,
    cfg: Dict[str, Any],
) -> Tuple[Optional[rest.Filter], Dict[str, Any]]:
    """Return a Qdrant filter built ONLY from indexed payload fields, plus a
    trace dict recording which constraints were applied and why."""
```

**Rules:**
- Only `doc_class`, `contains_table`, `entities`, `date_refs`, `access_tags`,
  `kind`, `tier` may appear in a filter. Anything else is a linear scan.
- Return `(None, trace)` when no constraint applies. Do **not** return a filter
  that matches everything — it costs time and proves nothing.
- The trace must say what was applied. A pre-filter that silently narrows to
  zero candidates is the single easiest way to fake a fast retriever.
- If the filter would eliminate every candidate, that is a result, not an error:
  record it in the trace and let the funnel fall through to unfiltered search.

**Tests required:**
- A filter is built from an entity constraint and contains exactly that field.
- An unindexed field never reaches the filter.
- `enabled: false` yields `(None, trace)` with `trace["skipped"] is True`.
- Mutation to prove: make `build_prefilter` always return `None` — at least one
  test must fail.

---

## 5. Task B — gated rerank (funnel stage 3)

**File:** `src/adaptive_retriever/funnel/rerank.py`

**Config already present:**
```yaml
funnel:
  rerank:
    mode: gated          # gated | always | never
    gate_score_margin: 0.15
    input_size: 100
    output_size: 5
```

**Build:**
```python
class Reranker:
    def __init__(self, model_name: str, device: str = "auto"): ...
    def score(self, query: str, texts: List[str]) -> List[float]: ...

def should_rerank(candidates, cfg) -> Tuple[bool, Dict[str, Any]]: ...
def apply_rerank(query, candidates, reranker, cfg) -> List[RetrievalCandidate]: ...
```

**Rules:**
- `Reranker.__init__` **raises** if the model will not load. No fallback, no
  constant scores. This exact bug already shipped once.
- `score` must return a list the same length as `texts`. Assert it. Never filter
  the list.
- Gate logic: rerank when the margin between the top fused score and the score
  at position `output_size` is **below** `gate_score_margin` — that is, when the
  ranking is not already confident. Record the computed margin in the trace so
  the gate's behaviour is auditable.
- `should_rerank` must be pure and testable with no model loaded.
- Reranking reorders and truncates to `output_size`; it must renumber
  `final_rank` from 1 and must not drop `metadata`.

**Tests required:**
- `should_rerank` returns False on a confident ranking, True on a flat one, with
  hand-computed margins.
- `mode: never` and `mode: always` both honoured.
- A length mismatch from `score` raises.
- Mutation to prove: make the gate always return True — a test must fail.

---

## 6. Task C — funnel orchestrator

**File:** `src/adaptive_retriever/funnel/pipeline.py`

```python
def run_funnel(
    query_text: str,
    vs: VectorStore,
    cfg: Config,
    profile: Optional[QueryProfile] = None,
    reranker: Optional[Reranker] = None,
) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
    """pre-filter -> hybrid RRF -> gated rerank. Returns candidates and a trace."""
```

**The trace is a deliverable, not debug output.** It must record, per stage:
input count, output count, elapsed ms, and the decision taken. This trace is
what makes the "adaptive" claim in the project title verifiable rather than
asserted. Write it to `reports/traces/` as JSONL.

**Rules:**
- Each stage is called through the functions above; no retrieval logic in this
  file.
- The funnel must work with `reranker=None` when `mode: never`.
- Do not catch exceptions to keep the pipeline running. A stage that fails must
  fail the query.

**Tests required:**
- All three stages called in order, verified with spies.
- Trace has an entry per stage with non-zero timings.
- `prefilter.enabled: false` skips stage 1 but still produces a trace entry
  marked skipped.

---

## 7. Task D — fixed baseline twin

**File:** `src/adaptive_retriever/funnel/baseline.py`

The baseline is the control in every comparison. It must differ from the
adaptive system in **exactly one dimension: adaptivity.**

- Same corpus, same chunks, same eval set, same embedding model, same top-k.
- Fixed `recursive_512` chunking, raw representations only, dense-only
  retrieval, no query transformation, no rerank, no escalation.
- It must read the **same index**. Do not build a second collection — that
  changes the corpus and invalidates the comparison.

Filter to `kind == "raw"` using the existing `kind` payload index.

**Tests required:**
- Baseline never calls the reranker, the transformer or the sparse search.
- Baseline and adaptive both read the same collection name.

---

## 8. Task E — metrics harness

**File:** `src/adaptive_retriever/eval/metrics.py` (extend if it exists)

**Metrics:** Recall@k, nDCG@10, MRR, plus latency p50/p95 and cost per query.

**Config already present:**
```yaml
eval:
  bootstrap_samples: 1000
  ci: 0.95
```

**Rules:**
- Every reported number carries a bootstrapped 95% CI. A point estimate with no
  interval is not a result.
- Items with `provenance: synthetic_rejected` are **excluded** from reported
  metrics and the excluded count is printed alongside every result. Never
  silently drop them.
- Bootstrap resamples **queries**, not individual judgements.
- Seed from `cfg.raw["seed"]` so runs are reproducible.
- Report baseline and adaptive side by side with the CI of the **difference**,
  not two separate CIs. Overlapping individual CIs do not mean no difference.

**Tests required:**
- Recall@k and nDCG against a hand-computed fixture.
- CI is narrower with 1000 samples than with 50.
- An excluded item changes the denominator and the printed excluded count.

---

## 9. Acceptance gate — all of these before Phase 2 is called done

```powershell
.\.venv\Scripts\Activate.ps1
ruff check --select F821,E9 src tests
python -m pytest -q
python check_server_mode.py
python check_hybrid.py
python -m adaptive_retriever.cli funnel "what methods were used to evaluate results" --trace
```

Required outcomes:
- ruff clean.
- All tests pass, and the new tests each fail under a stated mutation.
- `check_server_mode.py`: 3 PASS, `ingest_config_hash` still `a5d51992`.
- `check_hybrid.py`: all PASS, no duplicate chunks, text is raw text.
- A trace file written to `reports/traces/` showing three stages with real
  counts and timings.

---

## 10. Reporting format

When you finish a task, report exactly this and nothing more:

1. Files created or changed.
2. Tests added, and for each new behaviour, the mutation you injected and which
   test caught it.
3. Command output of the acceptance gate, trimmed to the verdict lines.
4. Anything you could not verify, stated plainly as unverified.

Do not report success for a step you did not run.
