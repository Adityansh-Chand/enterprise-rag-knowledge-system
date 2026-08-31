# ADR-001 — Four real retrievers behind one interface

**Status:** Accepted · **Date:** 2026-04 · **Supersedes:** the `EMBEDDING_PROVIDER` stub

## Context

The service claimed "hybrid semantic and lexical retrieval". It had one retrieval path:
vectors built by hashing tokens into 128 buckets, with cosine similarity over those
buckets. Cosine over hash buckets measures token overlap, so the advertised blend of
`0.65 * semantic + 0.35 * lexical` was combining a lexical signal with another lexical
signal. The "semantic" component was a five-entry hand-written synonym dictionary, and
one of the two evaluation queries was written to match an entry in it.

There was also a `EMBEDDING_PROVIDER = "local"` setting that raised on every value except
`"local"` — configuration shaped like a plug point, with nothing that could be plugged in.

The question was not "how do we make the claim true" but "what makes a pluggability claim
checkable at all".

## Decision

Define one `Retriever` protocol — `index(chunks)` / `search(query, k)` — and implement
**four** genuinely different methods behind it: BM25, LSA (TF-IDF → SVD, fitted on the
corpus), a sentence-transformers bi-encoder, and reciprocal-rank fusion over one lexical
and one semantic retriever.

`RETRIEVER` selects between them. An unrecognised value fails with an error naming the
supported ones.

## Alternatives considered

**Keep one retriever, correct the README.** Cheapest and honest. Rejected because the
interesting result in this repo is the *disagreement* between methods, and a single
retriever cannot produce a comparison table. The per-query-type breakdown is the finding;
one method makes it unobtainable.

**Ship the interface with one implementation and call it extensible.** This is what the
`EMBEDDING_PROVIDER` stub already was. An interface with one implementation has never had
its abstraction tested, and the odds it fits a second implementation unchanged are poor.
Four implementations is the smallest number that proves the seam is real — and it did
force changes: `index()` had to accept a corpus rather than stream, because LSA must fit
on the whole thing before it can transform anything.

**Retire the synonym dictionary but keep hash embeddings.** Rejected as the same failure
in quieter clothing.

## Consequences

- The default path now pulls a neural model, which costs a download on a cold clone and
  a cache layer in CI and Docker. Mitigated, and the size is documented in the README so
  a reviewer is not surprised.
- Relations like `chargeback ~ dispute` now emerge from corpus co-occurrence instead of
  being hand-written, so they cannot be quietly authored to make an evaluation pass.
- The interface is the substrate later work plugs into: adding ColBERT, SPLADE or HyDE
  becomes "implement `Retriever`, rerun the harness".
- Four retrievers means four sets of numbers to keep current, and BEIR runs are slow
  enough that they are deliberately not in CI.

## Revisit when

A retriever needs to stream or index incrementally, which `index(chunks)` cannot express.
That is a real limit of this interface, not an oversight — it was chosen knowing the
corpus fits in memory.
