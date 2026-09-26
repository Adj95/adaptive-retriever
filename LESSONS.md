# Lessons

A running record of what went wrong while building this system, why it went wrong, and the
rule that came out of it.

It exists because of one observation that kept repeating: **almost nothing in this project
failed loudly.** Bugs did not crash. They returned plausible values, printed success, and
moved on. Every single defect below was found by measuring something, never by reading an
error message.

---

## 1. Silent success is the default failure mode

### Hash bytes as embeddings

The first working version wrote SHA-256 digest bytes as "dense vectors" and
`hash(word) % 10000` buckets as "sparse vectors". Ingestion succeeded. The index filled.
Queries returned results ranked by score. Everything looked fine.

It was caught by measuring cosine similarity on a pair of sentences that mean the same thing
and a pair that don't. The paraphrase pair scored **0.0044**; the unrelated pair scored
**0.029**. The vectors carried no semantic signal at all, and the sparse indices weren't even
stable between processes.

> **Rule:** an embedding pipeline is not verified by "it ran". Measure similarity on a known
> pair before trusting an index.

### Fallbacks that hide the failure they're handling

Three components degraded quietly instead of failing: the embedder fell back to a stub when
the model wouldn't load, the reranker returned `[1.0] * n` on load failure, and hybrid search
fell back to dense-only when sparse was unavailable. Each one produced a system that ran
end-to-end while doing nothing useful.

The worst instance came last. With `vectorstore.mode: server` configured and the Qdrant
container not running, the code silently opened an **embedded local store** and kept going.
It ran a full four-arm experiment for thirty minutes against the wrong backend — 14 s/query
instead of 1.4 s/query, with payload indexes inert — and would have produced a complete
results table with confidence intervals that nobody could tell was wrong.

> **Rule:** in a measurement system, a fallback is a lie. If a component cannot do its job,
> raise and name what is missing. The cost of a crash is minutes; the cost of a plausible
> wrong number is every conclusion built on it.

### Errors swallowed on the way out

- An exception handler that caught the exception it had just raised.
- A `NameError` swallowed so thoroughly that `parse_cache: {hits: 428, misses: 0}` was
  printed as a fact while nothing was being cached.
- `[r for r in responses if r is not None]` applied to a batch result — silently changing
  its length and misaligning every response from its input.
- A missing `continue` after a parse failure, so failed parses fed downstream.

> **Rule:** any filter applied to a batch result must preserve length or raise. A cache
> statistic that has never been wrong is a statistic nobody is computing.

### Prompt generation cache keys must include context content, not just candidate IDs

In Experiment 006, `_build_cache_key` hashed `(query_text, tuple(candidate_ids))`. However, post-retrieval transformations like candidate text compression alter candidate context text while keeping candidate chunk IDs identical, and candidate reordering changes the prompt order. As a result, Arms 2, 3, and 4 hit Arm 1's cached prompt completion and returned instant (~1.4 ms) cached responses instead of generating distinct uncached answers.

> **Rule:** a generation cache key must include a hash of the full rendered `context_str` (or prompt template), not just chunk identifiers. Compression alters text and reordering alters prompt order — neither changes chunk IDs.

### Oversized chunks create retrieval evaluation artifacts

When documents lack paragraph boundaries (`\n\n` or `\n`), a chunker without a hard character-count fallback can emit chunks exceeding 1,000,000 characters. Sparse search (BM25) indexes the full chunk, while dense embedding models (BGE-small) and cross-encoder rerankers truncate at 512 tokens (~2,000 characters). As a result, hybrid arms exploit text beyond character 2,000 that dense baseline arms cannot see.

Under truncation-aware scoring (restricting `is_relevant()` to the first 2,000 characters), the headline improvement is roughly halved and no longer clears Bonferroni correction at $n=169$:
- Recall@10: $+0.1036 \rightarrow +0.0473$ ($p=0.0506$)
- nDCG@10: $+0.1642 \rightarrow +0.0542$ ($p=0.0378$)
- MRR: $+0.0935 \rightarrow +0.0483$ ($p=0.0198$)

About half the measured gain is BM25 exploiting oversized chunks; the remainder is positive across all three metrics but underpowered at $n=169$.

> **Rule:** chunkers must enforce a hard character-count cap matching the context window limit of downstream encoders (2,000 characters for 512-token models). Evaluate relevance strictly against the portion of text downstream models actually read.

---

## 2. Measuring the wrong unit

Two bugs shared a shape: the code was correct about something that wasn't the question.

**Fusion ranked representations, not documents.** Each chunk is indexed several times — raw
text plus generated summaries, questions and keywords — so 12,074 chunks became 22,422
vectors. Reciprocal Rank Fusion was keyed on representation id, which meant one chunk could
occupy several slots in a top-10 and Recall was being measured against a list containing
duplicates.

The fix also mattered in its ordering: representations collapse to their chunk **before**
fusion, not after. Fuse first and merge later, and a chunk with four representations
accumulates four RRF contributions against a single-representation chunk's one — ranking
documents by how often they happen to be indexed rather than by relevance.

**The returned text was the summary, not the source.** Each point stores its own
representation's text. A chunk retrieved because its LLM-written summary matched carried
that summary into the candidate. Left alone, the generator would have quoted and cited
model-generated text as source evidence — the worst available failure for a system whose
premise is grounded citation.

> **Rule:** name the unit you are ranking and the unit you are returning, and write a test
> that fails if they drift apart.

---

## 3. Config that looks wired but isn't

Four times, a setting was read, carried into a trace, and never applied to anything.

- `funnel.prefilter.max_candidates` was read and reported in the trace while capping
  nothing. Worse than unused: a trace that reports an unenforced limit is a trace that lies.
- `class_confidence` was presented as a computed confidence score. Measured against ground
  truth, it averaged 0.878 on correct predictions and 0.871 on wrong ones — no signal at all.
- `corpus_vocab_overlap` was hardcoded to `0.5`, while `vocab_overlap_threshold: 0.35` sat in
  the config waiting to compare against it.
- `gate_score_margin` — see below, the worst of the four.

> **Rule:** every config key is either enforced or deleted. A key that is read but has no
> effect is more dangerous than one that is never read, because it reads as working.

### The gate that could never close

`funnel.rerank.gate_score_margin: 0.15` was compared against the absolute difference between
fused scores. With `rrf_k = 60` and weights summing to 1, the maximum possible RRF score is
`1 / 61 = 0.0164` — an order of magnitude below the threshold. The condition
`margin < 0.15` was therefore **true for every query that would ever run**. "Gated" reranking
was unconditional reranking, and the trace reported a plausible margin the whole time.

The fix was a relative margin, `(top - cutoff) / top`, which is scale-free and survives
changes to `rrf_k`, the weights, or the number of fused sources.

> **Rule:** before comparing a threshold to a score, work out the score's range on paper. A
> threshold outside it is a constant in disguise.

---

## 4. The evaluation set was the least trustworthy component

Three separate integrity failures, in the one artifact every number is measured against.

**Fabricated verification.** `eval-verify` stamped `provenance: synthetic_verified` on 50
items without showing a single one to a human. The giveaway was that the marked items were
`rows[:50]` — contiguous from the start of the file, which no human review would produce.

**Unanswerable multi-hop items.** The generator paired each multi-hop query with one relevant
passage and one sampled at random. Retrieval could never return the random passage, so 28
items were silently depressing Recall for a system that was answering them as well as anyone
could. They were excluded with a recorded rejection reason rather than deleted.

**Class labels assigned by rota.** `query_class_expected` turned out to be the generator's
loop counter, not a property of the query. The label sequence
`aggregation, filterable, exact_lookup, conceptual, compound, multihop` repeats verbatim
through the file, and the twelve queries beginning *"what is the total"* carry **five
different** expected classes.

That last one invalidated a measurement in the other direction: the query classifier scored
18.3% against a 20.1% majority baseline, which looked like a broken classifier and was
actually a broken grader.

> **Rule:** audit the evaluation data as adversarially as the code. Check it for contiguity,
> for periodicity, and for identical inputs carrying different labels. And when an item is
> unusable, exclude it with a recorded reason — never delete it, and never quietly keep it.

---

## 5. Statistics that mislead

**A median on a bimodal distribution is not a trend.** Rerank cost per query is either ~0 ms
(gate skipped) or ~1,250 ms (gate fired). Sweeping the gate threshold, the *median* total
latency jumped from 208 ms to 1,345 ms between two adjacent settings, which looked like a
6.5× performance cliff and was used to justify a config choice. It was nothing of the kind —
the median simply crossed from one mode to the other as the fire rate passed 50%. Mean cost
scaled perfectly linearly with fire rate the whole time.

**Overlapping marginal intervals answer nothing.** Comparing two arms by checking whether
their individual confidence intervals overlap is not a test. A paired bootstrap that
resamples the same queries across both arms gives a much tighter interval on the difference —
in this project, the paired difference interval (width 0.112) was narrower than either arm's
own interval (width ~0.136), which is exactly the correlation you are meant to exploit.

**Selecting on 18 comparisons and reporting one.** A threshold sweep ran 6 settings × 3
metrics at 95% confidence, and the winner was chosen *because* its interval excluded zero.
At α = 0.05 you expect roughly one false positive in 18 by chance. Under Bonferroni
correction, nothing survived. The setting was kept, but the report now says plainly that it
is chosen on point-estimate efficiency and is **not** statistically confirmed.

> **Rule:** count your comparisons before you run them, report the count, and never call a
> selected winner confirmed.

---

## 6. Tests that cannot fail

A green suite proved surprisingly little.

- A test asserting that a config value reached the fusion used a mock similarity score of
  `0.9` against a weight of `0.9`. It would have passed whichever the code used.
- No test loaded the real `config/default.yaml`, so "this setting is wired up" was never
  actually demonstrated. The test file imported `Config` and never used it.
- A test encoded a *wrong belief* as an assertion — that changing the collection name must
  change `ingest_config_hash`. As written, every assertion in that test would have passed if
  the function returned a constant.
- 48 passing tests coexisted with a syntax error in `eval/build_set.py`, found in one pass by
  `ruff --select F821,E9` alongside 39 undefined names.
- Early mutation "proofs" used a blanket `return None`, which failed every test in the file
  and demonstrated only that the function still returned a value.

> **Rule:** every new behaviour gets a test proven to fail when that behaviour is removed, and
> the proof is a **targeted one-line mutation** naming the line changed and the single test
> that caught it. Include a positive control: at least one assertion that fails if the
> function under test returns a constant.

---

## 7. Reporting discipline

Two softer failures, both worth naming because they are easy to miss.

An acceptance criterion was **restated so the result would pass** — "Run 1 within 20% of
Run 3" quietly became "Run 2 within 20% of Run 3" in the summary, while Run 1 still missed by
4.4×. And a warm-up routine that *loaded* three models was reported as warming them; loading
a model does not create the CUDA context, move weights to the device, or initialise the
tokenizer. Only a throwaway inference does. Until that was fixed, the first real query still
paid 13 seconds of hidden start-up cost inside the measured stage timings.

> **Rule:** if a criterion fails, report the failure and the size of the gap. Never restate
> the criterion. And separate setup cost from work cost in any number you intend to call
> latency.

---

## 8. Two features that measurement removed

The point of building the harness was to be able to delete things.

**Reranking improves ordering but not coverage.** A sweep from never-rerank to always-rerank
moved Recall@10 by `+0.0059 [-0.0473, +0.0651]` — statistically flat. nDCG and MRR did move.
So the entire coverage gain over the baseline comes from hybrid multi-representation
retrieval, and the cross-encoder only reorders what retrieval already found.

**Per-class fusion weights add nothing.** Four arms on the same 169 queries: per-class routing
versus fixed `0.5/0.5`, `0.8/0.2` and `0.3/0.7` applied to every query. Fixed `0.5/0.5`
against per-class routing came back at `+0.0000` on Recall@10 (p = 1.000), `+0.0016` on
nDCG@10, `+0.0048` on MRR. None of the nine comparisons survived correction. A subsystem was
removed as a direct result.

> **Rule:** build the measurement before the feature, and be willing to delete the feature.

---

## 9. Implementation and Comparison Discipline

### 1. Three of four "this technique is harmful" findings were implementation bugs

HyDE −0.196 $\to$ −0.003 once routing was fixed. Decomposition −0.204 $\to$ −0.010 once the original query was retained.

> **Rule:** before publishing a large negative effect, check the implementation against the technique's definition.

### 2. A number without a comparison is not evidence

18.3% classifier accuracy (vs 20.1% majority), 0.74 cosine (vs 0.56 null), 79.88% 4-gram coverage (vs 12.43% shuffled), ROC AUC 0.61 (vs 0.50 chance). Each looked like a finding until the control was run.

> **Rule:** every published metric must be reported alongside its random or default control.

### 3. The wrong interpreter is invisible

Anaconda's global python with a CPU-only torch produced a 7x slowdown that was reported as a hardware bottleneck.

> **Rule:** every experiment script must print `sys.executable` and `torch.version.cuda` in its header. Never infer execution device or hardware performance from timing counters alone.

### 4. A config value read at the call site is not the config value in the file

Experiment 006 ran compression at a threshold that did not match `config/default.yaml`, and preflight did not print it.

> **Rule:** every config key that changes a published number must be printed by `check_preflight.py`.

### 5. Multi-representation indexing at partial coverage: no resolvable effect

At 8.75% corpus coverage (3,483 of 39,814 chunks), LLM-generated summaries, questions and keywords produced no change this evaluation can resolve. On the 32 queries whose gold passages were covered, all four moving metrics improved (Recall@5 +0.0625, nDCG@5 +0.0469, nDCG@10 +0.0263, MRR +0.0330) but none reached significance; n=32 cannot resolve an effect of this size, and roughly 104 powered queries would be required (174 under Bonferroni). On all 169 queries the multi-representation arm was slightly worse (Recall@10 -0.0237, p = 0.0428), which does not survive Bonferroni at alpha = 0.005. Slot competition was proposed as the mechanism, tested (did a representation enter top-10 while a gold raw chunk left?) and NOT confirmed. Representations are not near-duplicate vectors: median cosine to source is 0.79-0.82, only 5.55% at cosine >= 0.90.

> **Rule:** report what the data resolves. A small negative that fails multiplicity correction, with a mechanism that failed its own test, is a null - not a finding.

### 6. Small LLM keyword generation is unviable

Keyword generation on a 0.5B model produced malformed output in roughly half of cases (154 of 304 keywords, 50.7%, exceed 40 tokens; P50 is 41, max 169); it is 4.2% of representations so the effect on results is small, but the generator is not fit for that representation type.

---

## What actually caught these

Not the test suite, and never an exception. In order of how much they found:

1. **Measuring a quantity the code claims to produce** — cosine similarity on a known pair,
   accuracy against ground truth, a threshold compared to its score's real range.
2. **Arithmetic on paper.** The gate bug was a division; the latency cliff was a property of
   the median; the classifier result was a comparison against a majority baseline.
3. **Targeted mutation.** Removing a behaviour and checking that exactly one test notices.
4. **Auditing the data, not just the code.** Contiguity, periodicity, and identical inputs
   carrying different labels.
5. **`ruff --select F821,E9`**, which found a syntax error 48 green tests had missed.




---

## 10. Cascade Router Cancellation (Task R-02 Closure)

1. Retrieval is 1.0% to 4.5% of end-to-end query cost (75.76-303.32 ms out of 803-6,772 ms); tiering retrieval alone saves -0.51%.
2. The extractive tier (Tier 0) is 66.28x cheaper than Tier 2 (75.76 ms vs 5,021.43 ms) but succeeds on 37.87% of queries, capping a perfect oracle router at 36.06% maximum cost saving against Always-T2 (52.59% against Always-T3) across all queries, and 69.37% across answerable queries.
3. The best escalation signal is ROC AUC 0.6124 - identically the Phase 5 confidence score, already measured and rejected. No new signal exists.

> **Rule:** If end-to-end query latency is dominated by LLM answer generation (95%+), tiering retrieval alone cannot yield meaningful cost savings. A cascade router requires both an extractive fallback tier and a novel gate signal that beats pre-existing query confidence.

---

## 11. Distractor Sampling & Benchmark Ceiling Effects (Task R-06c-prep / R-06c-1)

Published HotpotQA distractor benchmarks use **hard negatives** selected via TF-IDF / BM25 lexical similarity. Sampling distractors at random from full Wikipedia yields **easy negatives** that vector embedding models (BGE-small) easily distinguish from gold documents (dense nDCG@10 = 0.9201 on 20k pool, 0.8687 on 100k pool). Expanding a random distractor pool adds negligible retrieval difficulty and cannot create metric headroom for component testing.

> **Rule:** Benchmark corpus subsampling must preserve published hard-negative mining strategies. Randomly sampled distractor pools create trivial retrieval tasks where baseline performance hits a ceiling (>0.85 nDCG@10) and components cannot demonstrate gain.

