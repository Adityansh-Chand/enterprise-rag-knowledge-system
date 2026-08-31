# ADR-002 — Dense retrieval as the default, not hybrid fusion

**Status:** Accepted · **Date:** 2026-04

## Context

Hybrid retrieval is the conventional answer: fuse a lexical and a semantic ranker and
take the best of both. It is what the service originally claimed to do, and it is what
most reference architectures recommend.

Measured on both corpora, unweighted fusion **lost to its own dense component**:
nDCG@10 0.7913 against 0.8577 on the synthetic corpus, 0.3423 against 0.3727 on
BEIR/NFCorpus. Fusion was destroying signal rather than adding it.

Since an unweighted blend is an arbitrary choice, `training/tune_fusion.py` swept the
lexical/semantic balance, splitting queries into a **dev** half for selection and a
**report** half that the sweep never sees. Both corpora independently chose the same
weight: semantic 1.0, lexical 0.0.

The data was asked which blend was best and answered: don't blend.

## Decision

Default to pure dense retrieval. Keep `hybrid` implemented, configurable and benchmarked,
and report that it loses.

## Alternatives considered

**Ship hybrid as the default anyway, because it is the expected architecture.** Rejected
on the plainest possible grounds: it is worse on every corpus measured, and the only
argument for it is that reviewers expect to see it. Shipping a slower, worse default to
look conventional is the exact failure this repository was rebuilt to remove.

**Tune the fusion weights until hybrid wins.** The sweep is exactly this, done honestly,
and the honest version chose the endpoint. Sweeping on the report half would have found
something better-looking and meaningless.

**Delete the hybrid retriever.** Rejected. The null result is worth more than the code
costs, and removing it would leave the README asserting that fusion loses with nothing in
the repo able to demonstrate it.

**Per-query routing instead of one global weight.** Not an alternative — a successor.
See the routing work: choosing per query beats the global choice (0.9458 against 0.9390),
because one weight cannot exploit BM25 being genuinely better on *some* queries. That
result only became visible after fusion was measured properly.

## Consequences

- The default path is the slow one. Dense query latency is roughly 6× BM25 on
  BEIR/NFCorpus, and startup requires a model. `docker-compose` therefore runs `bm25` so
  the demo starts instantly, with the stronger setting documented.
- The repo carries a benchmarked component that is not recommended, which needs
  explaining every time someone reads the retriever list.
- The finding generalises poorly and is stated that way: fusion losing on two corpora is
  evidence about these corpora, not a claim about hybrid retrieval in general.

## Revisit when

The corpus grows enough that first-stage dense recall degrades, or a query mix arrives
with substantially more exact-identifier traffic — the one regime where BM25's advantage
is large and a blend could pay. Re-run `tune_fusion.py`; the machinery to answer it is
already there.
