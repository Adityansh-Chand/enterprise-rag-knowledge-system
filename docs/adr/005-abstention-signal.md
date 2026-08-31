# ADR-005 — Serve the crudest abstention signal, calibrated on positives only

**Status:** Accepted · **Date:** 2026-08

## Context

Retrieval always returns its top *k*. Ask the corpus something it cannot answer and it
returns the *k* least-bad documents with no indication that they are wrong, and the
extractive generator quotes one of them. The resulting answer is **perfectly grounded and
completely wrong** — groundedness 0.96 alongside a hallucination rate of 0.67, measured.

Groundedness cannot see this failure, because the answer really was copied out of a
retrieved chunk. Something else has to decide when not to answer.

Three signals were implemented and measured against 12 held-out unanswerable queries,
scored as AUC:

| signal | bm25 | dense |
|---|---|---|
| `top_score` — the raw top retrieval score | **0.6937** | **0.7722** |
| `coverage` — query-term overlap with the best sentence | 0.6292 | 0.6111 |
| `margin` — how far the top result stands out from the rest | 0.5549 | 0.6465 |

## Decision

Serve `top_score`, **for dense retrieval only**. Fit its threshold per retriever at the
**5th percentile of the answerable distribution, using no unanswerable examples**. Keep
`coverage` and `margin` implemented and benchmarked, and publish the operating curve
alongside the chosen point.

### Why BM25 is fitted but not served

BM25 has the second-best AUC of anything measured, and acting on it would have been a
mistake. It was caught by the flagship demo query — `ERR-4021 remediation steps`, which
BM25 answers **perfectly** (nDCG@10 1.0000 on exact-identifier queries) — being declined.

The cause is IDF. That answer is duplicated across fifteen near-identical runbooks, one
per service and region, so the term is common and the score is low. Across the answerable
set:

| answers appear in | mean BM25 top score |
|---|---|
| many documents (n=20) | **4.116** |
| a single document (n=40) | **11.332** |

The 5th-percentile threshold of 1.687 lands squarely on the exact-identifier block — the
queries BM25 is *best* at. A raw BM25 score is not comparable across queries: it mixes
"how well was this matched" with "how rare are these terms", and abstention needs only
the first. Dense cosine similarities are bounded and normalised, so they carry the same
meaning from one query to the next, which is why the same signal is usable there.

The AUC of 0.6937 hid this completely, because the unanswerable queries happened to score
low too. **A signal can rank well and still be the wrong thing to threshold on.** The
artifact therefore separates fitting from serving: BM25's threshold is recorded, and
`abstain: false` keeps it out of the serving path.

## Alternatives considered

**`coverage` — lexical overlap between query and best sentence.** The intuitive choice,
and structurally wrong for this system. The corpus deliberately contains
`vocabulary_mismatch` queries that share *no content word* with their own relevant
document — precisely the queries dense retrieval exists to win. A lexical confidence
signal scores them near zero whether or not an answer exists, so a threshold filtering
unanswerable questions would also refuse the questions the system is best at. Measured at
0.61–0.63 AUC, barely above chance, and much of what it does capture is stopword overlap.

**`margin` — peakedness of the top result.** Scale-free, so one threshold could serve any
retriever, which was the appeal. Near-random in practice (0.55 on BM25): BM25 scores whole
groups of answerable queries at exactly zero, collapsing the left tail, and dense cosines
are too tightly packed for the gap to carry information.

**Sweep the threshold to maximise F1 across both query sets.** Rejected as circular. The
12 unanswerable queries are the only held-out test of abstention that exists; a threshold
chosen on them has already seen the answer, and the reported hallucination rate would
measure nothing. Calibrating from the positive distribution alone is the same discipline
the incident service uses for alert budgets, and for the same reason: production rarely
supplies labelled negatives, but always supplies the positives.

**A learned abstention classifier over all three signals plus retrieval features.** The
right next step and deliberately not taken yet. With 12 negatives it would fit noise, and
generating more from the same generator would produce negatives shaped like the ones it
already sees.

**Do not abstain at all; report the hallucination rate and stop.** Rejected — the failure
is measurable and partly addressable, and measuring without acting would be a worse
answer than acting imperfectly and saying so.

## Consequences

- The system declines about 5% of questions it could answer, by construction, and that is
  the price named up front rather than discovered.
- Abstention is off under the `bm25` default that `docker-compose` uses for fast startup,
  so the demo does not exercise it. Stated rather than papered over by switching the
  default to a retriever that needs a model download.
- **At the served point it still answers roughly three quarters of the questions it should
  refuse.** This is a weak defence, published as one. The five lexical features available
  do not contain a strong signal, and pretending otherwise would be the failure this
  repository exists to avoid.
- The operating curve is published because the AUCs mislead: dense separates better
  (0.77 vs 0.69) but at the served point catches *fewer* unanswerable queries (0.25 vs
  0.33). BM25's curve is flat — 6× the answerable loss buys nothing — while dense's is
  actionable, reaching 0.83 at 40% loss. **Better separation did not mean better
  behaviour**, and a single number would have hidden that.
- Thresholds are retriever-specific, so switching `RETRIEVER` without recalibrating gives
  an unfitted retriever. That case returns "never abstain" rather than a wrong threshold,
  which loses a defence rather than breaking the service.
- The check runs before the LLM call, so it doubles as cost control.

## Revisit when

Enough genuine negatives accumulate — real user queries the corpus could not answer — to
fit a classifier without it memorising a generator. That is the point at which the
learned version stops being a worse answer than the crude one.
