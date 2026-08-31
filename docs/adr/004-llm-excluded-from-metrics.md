# ADR-004 — No reported metric may come from the LLM path

**Status:** Accepted · **Date:** 2026-04 · **Reaffirmed:** 2026-08

## Context

The service has an optional LLM backend behind a provider-agnostic seam
(`llm/client.py`, roughly `complete(system, user) -> str`). Callers never learn which
provider is configured; `openai_compatible`, `anthropic` and `none` are the supported
values, and `none` is the default.

The obvious use is to generate answers with it and report the quality. That would produce
better-looking numbers than sentence extraction does, and it is what the title on this
kind of project usually implies.

## Decision

The LLM path may run. It may not be the source of any number in the README, any model
card, or any quality gate. Every reported metric comes from the local, deterministic path.

## Alternatives considered

**Report LLM-generated answer quality as the headline.** Rejected on reproducibility. The
number would depend on a vendor, a model version, a sampling temperature and a date. A
reviewer cloning the repo six months later, with a different key, would not reproduce it,
and could not tell whether a difference meant a regression or a model update. A metric a
reader cannot check is a claim, not a measurement.

**Pin a specific model and report that.** Better, and still rejected: pinned models are
deprecated and withdrawn, the pin needs a key to exercise, and CI cannot run it. The
result would be a headline number no automated check ever re-derives — the exact
condition under which committed metrics quietly drift away from the truth.

**Remove the LLM seam entirely.** Rejected. It is genuinely useful, it is honest as
built, and it costs nothing when unset. Cutting it to avoid the temptation would remove
capability rather than resolve the question.

**Report both, clearly labelled.** The nearest miss, and revisitable. Rejected for now
because two sets of numbers in every table invites exactly the comparison the reader
cannot verify, and because the harness needed to make the comparison meaningful is only
now built (ADR-005).

## Consequences

- The reported answer-quality numbers are those of an extractive generator, and they are
  modest: fact coverage 0.32, attribution 0.53. An LLM would very likely score better.
  That gap is stated in the README rather than closed by switching paths.
- The evaluation harness is nonetheless LLM-ready: `generation_bench.py` scores whatever
  the generator returns, so pointing it at the LLM path is one environment variable, not
  a rewrite. The capability is demonstrated; only the bill is absent.
- The abstention check runs *before* the model call, so the questions most likely to
  produce invention are also the calls not paid for.
- Anyone assuming "RAG" implies a generative model will find the opposite, and the README
  says so in the first section rather than in a footnote.

## Revisit when

A locally-runnable open model can be pinned by weights hash and executed in CI within its
time budget. That removes the vendor, the version drift and the key at once, and turns the
rejection above into an ordinary engineering question. Until then this stands.
