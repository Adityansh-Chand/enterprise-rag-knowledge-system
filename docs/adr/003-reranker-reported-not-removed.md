# ADR-003 — Ship the reranker as a measured null result

**Status:** Accepted · **Date:** 2026-04

## Context

The original `reranker.py` rescored candidates using the same token-overlap function the
retriever had already used to rank them. It could not reorder anything by construction —
it was a no-op wearing the name of a component.

Two replacements were built and benchmarked against no reranking at all:

- a **fitted pairwise model** — logistic regression over term overlap, query coverage,
  identifier match, title overlap, length and prior rank, trained on the relevance
  judgments
- a **pretrained cross-encoder**, as the strong external reference point

Both results were negative. The fitted reranker changes held-out nDCG@10 by **+0.0000**
on the synthetic corpus (−0.0003 over hybrid candidates). The cross-encoder buys
**+0.0011** for roughly **1116 ms** per query — about 250× the cost of the retrieval it
is correcting.

The cause is visible in the retrieval numbers rather than the reranking ones: first-stage
retrieval at depth 20 already reaches 0.8889 nDCG@10 on this corpus. There is nothing
left to reorder. The fitted weights are sensible — `title_overlap` +7.89,
`identifier_match` +4.01 — so the model did learn something real about relevance. It had
no mistakes left to correct.

## Decision

Keep both rerankers, benchmarked, and report the null result as the finding. Do not
enable reranking by default. `rerank()` returns the input order unchanged when no fitted
artifact is present.

## Alternatives considered

**Delete the reranker and drop the claim.** Rejected. "Reranking earns its keep when
first-stage retrieval is weak, and here it is not" is a more useful thing for a reader to
learn than silence, and it can only be said with the measurement in the repo.

**Enable the cross-encoder by default because it is technically ahead.** +0.0011 nDCG@10
is inside the noise of this corpus, and 1116 ms per query is not. Paying three orders of
magnitude in latency for a difference that cannot be distinguished from zero is a bad
trade, and stating the ratio is more informative than stating the improvement.

**Make the corpus harder until reranking helps.** This was considered explicitly and
rejected as the most tempting mistake available. It would have produced a repository
where reranking "works", by tuning the problem to fit the answer. The corpus is built to
exercise retriever disagreement; bending it to rescue a component would corrupt every
other number computed on it.

**Report only the cross-encoder and omit the fitted model.** Rejected. The fitted model
is the one that demonstrates the ML work, and its failure is more instructive than the
pretrained model's marginal success.

## Consequences

- The README carries a component described as not worth enabling, which reads oddly until
  the reasoning is read. Accepted deliberately.
- A reader gets a cost ratio (250×) rather than a bare improvement, which is the number
  that actually decides whether to deploy it.
- Both paths stay exercised in CI, so the null result is re-derived rather than remembered.

## Revisit when

First-stage retrieval degrades — a larger or noisier corpus, or a harder query mix.
Reranking is worth re-benchmarking the moment retrieval stops being near-ceiling, and the
harness to do it is already committed. The headroom, not the reranker, is what changed.
