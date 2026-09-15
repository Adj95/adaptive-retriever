# PHASE 3 — AMENDMENT 1 (2026-09-15)

Read this **after** PHASE3_BUILD_ORDER.md. Where the two disagree, this wins.

## Why

Phase 3 as originally written routes each transformation strategy by query class. Two
measurements have since removed the ground that design stood on.

1. **The query classifier cannot be measured.** `query_class_expected` in the evaluation set
   was assigned by a generation rota, not derived from query content — the label sequence
   `aggregation, filterable, exact_lookup, conceptual, compound, multihop` repeats verbatim
   through the file, and the twelve queries beginning "what is the total" carry five
   different expected classes. The classifier's 18.3% accuracy measures the grader, not the
   classifier.

2. **Per-class routing adds nothing where it could be measured.** Experiment 004 compared
   per-class fusion weights against fixed weights applied to every query, on the same 169
   items. Fixed `0.5/0.5` versus per-class routing: `+0.0000` Recall@10 (p = 1.000),
   `+0.0016` nDCG@10, `+0.0048` MRR. Zero of nine comparisons survived Bonferroni correction.

Routing a transformation by a class label is therefore unjustified — the label is
unmeasurable, and the one routed decision we could test contributed nothing.

## What changes

**Per-class routing is removed from Phase 3.** Transformations are applied and measured
**globally**, not per class.

| Original | Amended |
|---|---|
| §4 Task A — route by `query.transformation.mapping[query_class]` | Keep the interface, the cache and the `fixed` override. **Delete the per-class mapping.** A single `query.transformation.strategy` applies to every query. |
| §5 Task B — `self_query` fires only on `filterable` queries | **Attempt filter extraction on every query.** It is a no-op when nothing extractable is present. This also removes the problem that `filterable` was predicted on 5 of 169 queries. |
| §6 Task C — HyDE for `conceptual` | Apply to every query when enabled; measure globally. |
| §7 Task D — decompose for `compound` | Apply to every query when enabled; measure globally. |
| §8 Task E — per-class breakdown of results | **Drop the per-class breakdown.** Report four global arms: `none`, `self_query`, `hyde`, `decompose`, each as a paired difference against `none`. |

**Also remove from the codebase**, in the same pass as Task A's amendment:

- `funnel.hybrid.class_weights` — replace with a single `funnel.hybrid.weights:
  {dense: 0.5, sparse: 0.5}`. Delete the per-class weight selection in `query_hybrid`,
  and the test asserting every `QueryClass` member has an entry.
- `query.transformation.mapping` — replace with `query.transformation.strategy`.

**Keep the classifier itself.** `profile_query` still produces a `QueryFingerprint`, and its
`extracted_filters` still feed the pre-filter. It simply no longer routes anything. Record
`query_class` in the trace as an observation, not as a decision input.

## What this is not

This is not abandoning adaptivity. It is refusing to claim adaptivity that has not been
measured. If a global strategy shows a real effect, per-query routing becomes worth
revisiting — on measurable query features (backticked identifiers, clause count, corpus
vocabulary overlap), not on a six-way semantic label, and validated on a held-out split.

## Standing rules unchanged

§0 (the five rules), §2 (environment facts), §3 (failure catalogue) and §10 (reporting
format) of PHASE3_BUILD_ORDER.md all still apply in full.
