# Architecture Decision Records

Decisions that shaped this service, each with the alternatives that were actually
considered, the evidence that settled it, and what would make it worth revisiting.

A record is written when a choice was **contested** — when a competent engineer could
reasonably have gone the other way, and the reason it went this way is not recoverable
from reading the code. Choices with one obvious answer are not recorded; a directory of
records for things nobody would dispute is a directory nobody reads.

Records are immutable once accepted. A decision that changes gets a new record that
supersedes the old one, and the old one stays, because the reasoning that turned out to
be wrong is usually the more useful half.

| # | Decision | Status |
|---|---|---|
| [001](001-pluggable-retriever-interface.md) | Four real retrievers behind one interface | Accepted |
| [002](002-dense-default-over-fusion.md) | Dense retrieval as the default, not hybrid fusion | Accepted |
| [003](003-reranker-reported-not-removed.md) | Ship the reranker as a measured null result | Accepted |
| [004](004-llm-excluded-from-metrics.md) | No reported metric may come from the LLM path | Accepted |
| [005](005-abstention-signal.md) | Serve the crudest abstention signal, calibrated on positives only | Accepted |

Portfolio-wide decisions that apply to all five services live in
[`ai-engineering-portfolio/docs/adr/`](https://github.com/Adityansh-Chand/ai-engineering-portfolio/tree/main/docs/adr).
