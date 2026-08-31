# Enterprise RAG Knowledge System

A retrieval bench. Four retrievers behind one interface — BM25, LSA, a dense
bi-encoder, and reciprocal-rank fusion — evaluated by one harness against **real
public IR benchmarks with human relevance judgments**, plus a synthetic corpus
that keeps the demo runnable offline.

## What was wrong before

This repo previously claimed *"hybrid semantic and lexical retrieval"*, *"local
hashed embeddings"*, *"query-aware reranking"* and *"grounded answer generation"*.
Concretely:

- **The "semantic" half was lexical.** `embedder.py` hashed tokens into 128 buckets
  with SHA256. Cosine similarity over hash buckets measures token overlap — synonyms
  land in unrelated buckets. So `0.65 * semantic + 0.35 * lexical` blended one lexical
  signal with another.
- **The actual semantics were a 5-entry hardcoded dict** (`vacation → leave`,
  `wfh → remote`) — and one of the two eval queries was literally `"vacation days"`.
- **The reranker could not rerank.** It rescored candidates using the same
  `tokenize()` overlap the retriever had just used, so its ordering was a monotone
  transform of its input.
- **The generator did not generate.** It returned
  `f"Based on the knowledge base: {first_line_of_context}"` regardless of the question,
  and reported the top retrieval score under the name "confidence".
- **The evaluation could not fail.** The corpus was **4 sentences**; the eval was
  **2 substring-matched queries**; chunking was 3-sentences-with-overlap so the top
  chunk contained both expected strings whatever retrieval did.
  `rag/evaluator.py:precision_at_k` ignored `k` entirely.

All of it is replaced below, and the numbers are measured.

## Retrievers

One interface (`rag/retrievers/base.py`), four implementations:

| Retriever | What it is | Where it wins |
|---|---|---|
| `bm25` | Term frequency with saturation and length normalisation | Exact rare tokens — error codes, API paths, clause numbers |
| `lsa` | TF-IDF → `TruncatedSVD`, **fitted on the corpus** | Mild synonymy learned from co-occurrence |
| `dense` | `BAAI/bge-base-en-v1.5` bi-encoder | Paraphrase and vocabulary mismatch |
| `hybrid` | Reciprocal rank fusion over a lexical and a semantic retriever | Two genuinely independent signals |

`hybrid` fuses **ranks**, not scores — BM25 produces unbounded term-weight sums while
cosine sits in [-1, 1], so averaging them directly would let the larger numeric range
dominate for reasons unrelated to relevance.

## Results — synthetic mixed-enterprise corpus (108 docs, 60 queries)

This corpus exists so the demo runs offline. It is **not** a benchmark; its value is
the per-query-type breakdown, because it was built specifically to make the retrievers
disagree.

| Retriever | nDCG@10 | Recall@10 | MRR@10 | query ms |
|---|---|---|---|---|
| bm25 | 0.6820 | 0.6444 | 0.6708 | 0.4 |
| lsa | 0.6544 | 0.6222 | 0.6401 | 1.7 |
| **dense** | **0.8577** | **0.7944** | **0.8667** | 757 |
| hybrid (bm25+dense) | 0.7913 | 0.7367 | 0.7950 | 258 |

### nDCG@10 by query type — the actual finding

| Query type | bm25 | lsa | dense | hybrid |
|---|---|---|---|---|
| `exact_identifier` | **1.0000** | 0.9379 | 0.9219 | **1.0000** |
| `vocabulary_mismatch` | 0.1016 | 0.0370 | **0.6154** | 0.3752 |
| `paraphrase` | 0.9806 | 0.9700 | **1.0000** | **1.0000** |
| `polysemy` | 0.8712 | 0.8000 | **1.0000** | **1.0000** |
| `acronym` | 0.6667 | **1.0000** | 0.9537 | 0.7824 |

Three things worth reading carefully:

**BM25 wins exact identifiers outright and loses vocabulary mismatch almost
completely** (0.1016). That is by construction: the generator asserts every
`vocabulary_mismatch` query shares *no content word* with its target document, so BM25
provably cannot match. A test enforces that guarantee.

**Hybrid is worse than dense alone** (0.7913 vs 0.8577) — the opposite of what I
expected. Equal-weight RRF lets BM25's near-zero mismatch performance drag the fusion
down from 0.6154 to 0.3752, and mismatch queries are a third of the set. **Naive rank
fusion can underperform its own best component when the components are very unequal.**
This is not a quirk of the synthetic corpus — it **replicates on BEIR/NFCorpus**
(0.3423 vs 0.3727). See *Weighted fusion* below for what happened when the weight was
allowed to come from data.

**LSA sits slightly *below* BM25 overall** (0.6544 vs 0.6820). On a 108-document corpus
the latent space is too thin to learn much synonymy. It does win the acronym category
outright. LSA is genuinely semantic and genuinely fitted, and on a corpus this size that
is not enough to beat a good lexical baseline.

## Results — BEIR benchmarks (real human relevance judgments)

These are **real** IR benchmarks: publicly released corpora with human-assigned
relevance judgments. Unlike the synthetic corpus above, nothing here was written
by the same hand as the code being evaluated.

Source: the BeIR collection on the HuggingFace hub (`cc-by-sa-4.0`).

**NFCorpus** — 3,633 medical documents, 323 test queries with human qrels:

| Retriever | nDCG@10 | Recall@10 | MRR@10 | index (s) | query (ms) |
|---|---|---|---|---|---|
| bm25 | 0.2831 | 0.1326 | 0.4881 | 1.8 | 10 |
| lsa | 0.2613 | 0.1277 | 0.4244 | 5.2 | 20 |
| **dense** | **0.3727** | **0.1784** | **0.5662** | 2475 | 59 |
| hybrid (bm25+dense) | 0.3423 | 0.1637 | 0.5627 | 0.8 | 96 |

**The synthetic corpus's two main findings replicate on real, human-judged data:**

1. **Dense beats BM25** — 0.3727 vs 0.2831, a 32% relative improvement. NFCorpus
   queries are lay-language health questions against technical abstracts, which is
   vocabulary mismatch in its natural form, and it is exactly where the synthetic
   bench predicted dense would win.
2. **Hybrid is again worse than dense alone** — 0.3423 vs 0.3727. The same effect
   as on the synthetic corpus, now on a benchmark nobody here constructed. That
   makes it a much more credible observation than a single synthetic result:
   equal-weight reciprocal rank fusion genuinely can underperform its stronger
   component when the two are unequal. Weighting it from data fixes the
   underperformance but gains nothing beyond dense alone — see *Weighted fusion*.

LSA again lands slightly below BM25, consistent across both corpora.

Note the cost column: indexing with `bge-base` took **2,475 seconds** on CPU for
3,633 documents (~1.5 docs/s — these are long abstracts). Embeddings are cached
after the first run. BM25 indexed the same corpus in 1.8 seconds and answers in
10ms. Whether the ~9-point nDCG gain is worth ~1,400x the indexing cost is a real
engineering decision, not an obvious one.

**One caveat, stated rather than glossed:** this BM25 is `rank_bm25` with default
`k1`/`b`, no stemming and no domain stopword handling. Published NFCorpus BM25
numbers generally come from Anserini/Lucene with proper analysis and sit higher.
The comparison *between the retrievers here* is apples-to-apples — same tokenizer,
same harness, same qrels — but the comparison to published leaderboards is not.

`scifact` and `fiqa` are configured in `rag/beir_data.py` and run the same way:

```bash
python scripts/prefetch_beir.py --dataset scifact
python evaluation/harness.py --beir scifact
```

FiQA (57,638 documents) was not run here: at the measured CPU encoding rate it is
roughly a ten-hour job on this machine.

## Weighted fusion — the weight the data picks is "don't fuse"

Unweighted fusion losing to its own dense component is a fixable-looking problem, so
`training/tune_fusion.py` sweeps the lexical/semantic balance. The selection is kept
honest: queries are split into **dev** and **report** halves, the sweep only ever sees
dev, and the chosen weight is applied once to the report half.

Both corpora agree, and both pick the endpoint:

| Semantic weight chosen on dev | synthetic | BEIR/NFCorpus |
|---|---|---|
| | **1.0** | **1.0** |

Scored on the report half, which was never used for selection:

| | synthetic | BEIR/NFCorpus |
|---|---|---|
| bm25 alone | 0.6996 | 0.2775 |
| dense alone | 0.9390 | 0.3553 |
| hybrid, unweighted | 0.8551 | 0.3335 |
| hybrid, **tuned** | **0.9390** | **0.3553** |

**Weighting fixes the underperformance and gains nothing beyond it.** Tuned fusion beats
unweighted by +0.0839 (synthetic) and +0.0218 (NFCorpus), and then ties dense *exactly* —
because a semantic weight of 1.0 means a lexical weight of 0.0. The sweep's answer is
that BM25 should not be in the mix at all.

So the original finding was not a fusion bug. Fusion was including a component that,
averaged over these query sets, contributes nothing on top of dense.

**But averaged is doing work in that sentence.** BM25 scores **1.0000** on
`exact_identifier` queries against dense's 0.9219. It is genuinely better there and
genuinely useless elsewhere, and a single global weight cannot express that. The
promising direction is not a better weight — it is **routing per query**, sending
identifier-shaped queries to the lexical retriever and everything else to dense. That is
not implemented here, and no claim is made about it.

Raw sweep and both halves: [`models/artifacts/fusion_weights.json`](models/artifacts/fusion_weights.json).

## Per-query routing — where a global choice loses

Tuning weighted fusion chose **pure dense** on both corpora: semantic weight 1.0,
lexical 0.0. Honest, and also a limitation — one weight applied to every query
cannot exploit BM25 being genuinely better on *some* of them.

`ERR-4021 remediation steps` is a lexical problem: the answer contains that exact
token and nothing else does, so an embedding that places it near other error codes
actively hurts. `customers cannot complete purchases at the final step` is the
opposite — it shares no content word with the document about checkout latency, and
BM25 scores it **0.0000**.

So the choice is made per query, from the query text alone:

```python
a query containing an identifier-shaped token  ->  lexical
everything else                                ->  semantic
```

```bash
python training/tune_router.py
RETRIEVER=router uvicorn api.server:app --port 8000
```

**Report half only** (queries split dev/report by sorted id, the rule fixed before
measurement — the same discipline `tune_fusion.py` uses):

| retriever | nDCG@10 | Recall@10 | MRR@10 |
|---|---|---|---|
| bm25 | 0.6996 | 0.6667 | 0.6889 |
| dense | 0.9390 | 0.8467 | 0.9500 |
| **router** | **0.9458** | **0.8533** | 0.9500 |
| hybrid | 0.8551 | 0.8000 | 0.8611 |

**+0.0068 over pure dense** — small, and the per-type breakdown is what makes it
believable rather than noise:

| query type | bm25 | dense | router |
|---|---|---|---|
| exact_identifier | **1.0000** | 0.9710 | **1.0000** |
| acronym | 0.5000 | **1.0000** | **1.0000** |
| paraphrase | 0.9692 | **1.0000** | **1.0000** |
| polysemy | 0.6781 | **1.0000** | **1.0000** |
| vocabulary_mismatch | 0.0000 | **0.7676** | **0.7676** |

The router takes BM25's win on identifier queries and dense's win everywhere else.
That is the whole mechanism, and it is why the aggregate gain is small here: only
about a quarter of these queries are identifier-shaped. On a corpus with more of
them the same rule pays more.

### Two constraints that make this believable

**The rule never sees the corpus's query-type labels.** They exist in
`datasets/queries.json`, and routing on them would score beautifully and mean
nothing — it is exactly the circular evaluation this repository was rebuilt to
remove. A test asserts the rule fires *only* on `exact_identifier` queries, which
is the check that would catch a rule tuned to fire indiscriminately.

**An acronym alone does not trigger it.** `what does our DPA say about
sub-processors` is prose that happens to contain an acronym, and the document
answering it uses the expansion — BM25 scores 0.5000 there against dense's 1.0000.
A cruder rule firing on any uppercase run would have routed it the wrong way.

Routing costs nothing at query time (a regex) but indexes both retrievers, so
memory and index time are the sum of the two. That is stated rather than elided.

## Reranking — a measured null result

`rag/reranker.py` is a **fitted** pairwise model (logistic regression over term
overlap, query coverage, identifier match, title overlap, length and prior rank),
trained on the corpus relevance judgments by `training/train_reranker.py`.

On the synthetic corpus it changes held-out nDCG@10 by **+0.0000** — it reorders
nothing. (Trained over hybrid candidates instead, the delta is −0.0003. Same story.)

The reason is visible in the numbers: retrieval at depth 20 already reaches 0.8889
nDCG@10 on this corpus, so there is essentially nothing left to reorder. Reranking earns
its keep when first-stage retrieval is weak; here it is not.

The fitted weights are sensible — `title_overlap` +7.89, `identifier_match` +4.01 — so
the model learned something real about relevance. It simply had no mistakes left to
correct.

This is reported rather than tuned away, and the README claim is worded to match the
result. `rag/reranker.py` returns the input order unchanged when no fitted artifact is
present — an honest no-op is better than a fake improvement.

### Was the fitted reranker simply too weak?

That is the fair objection to a null result, and it needed answering: "reranking
does not help here" and "our reranker is bad" have opposite implications. So a
pretrained **cross-encoder** — `ms-marco-MiniLM-L-6-v2`, Apache 2.0 — was benched
as the strong reference point. It reads query and document together through a
transformer rather than comparing independent representations, which is exactly
the signal a bag of features cannot see.

```bash
python evaluation/rerank_bench.py --beir nfcorpus
```

**BEIR/NFCorpus, human relevance judgments, dense retrieval, depth 20:**

| Arm | nDCG@10 | Δ | MRR@10 | ms/query |
|---|---|---|---|---|
| no reranking | **0.3727** | — | 0.5662 | 0.0 |
| fitted pairwise | 0.3664 | −0.0063 | 0.5702 | 4.5 |
| cross-encoder (MiniLM-L-6) | 0.3738 | **+0.0011** | **0.5847** | **1116.4** |

On our synthetic corpus the cross-encoder is *worse* (−0.0174).

**The question is answered: the fitted reranker was not the problem.** A strong,
purpose-built cross-encoder buys **+0.0011 nDCG@10 for 1.1 seconds per query** —
roughly 250× the cost of the fitted model for a difference indistinguishable from
noise.

One honest nuance in the other direction: MRR@10 does improve, 0.5662 → **0.5847**.
The cross-encoder genuinely does reorder the very top of the list better. It just
does not move nDCG@10, and Recall@10 cannot move at all — reordering a fixed
candidate list cannot add documents to it, which is asserted as a sanity check on
the harness itself.

**So it is benched and not shipped.** The cross-encoder is not in the serving
path: paying a second per query for +0.0011 would be a worse decision than the
null result it was brought in to test. Deciding *not* to ship a component after
measuring it is the point of measuring it.

## Answer quality — groundedness is not correctness

Retrieval metrics stop at "did the right document come back". `evaluation/generation_bench.py`
measures what happens after that, against 60 answerable queries with gold content units and
**12 held-out questions the corpus cannot answer**. Every judgment is computed against the
corpus, never by asking a model to grade itself — which is why it runs offline, on the
extractive path, with no API key and no spend.

| metric | bm25 | dense | what it asks |
|---|---|---|---|
| groundedness | **0.9593** | **0.9611** | is the answer traceable to retrieved text |
| fact_coverage | 0.3889 | 0.4083 | does it say the thing that mattered |
| attribution | 0.5500 | 0.6056 | did the sentence come from a *relevant* chunk |
| context_utilisation | 0.3533 | 0.3100 | how much retrieved text reached the answer |
| **hallucination rate** | **0.6667** | **0.7500** | answered anyway when it could not know |

**The finding: groundedness reads 0.96 next to a hallucination rate of 0.67.** Both numbers
are correct. An extractive generator copies sentences, so it is grounded by construction —
and when the question is unanswerable, retrieval still returns its top *k*, and a perfectly
grounded sentence from a genuinely irrelevant document is exactly what comes out. Groundedness
is worth reporting and worthless alone. It is the metric most likely to be quoted as evidence
of safety, and on this system it certifies almost nothing.

Fact coverage says the rest: under half the content that actually answers the question makes
it into the answer, and roughly two fifths of the answer sentences come from a chunk that was
never relevant. On `vocabulary_mismatch` queries attribution falls to **0.0351** under BM25 —
it retrieves the wrong document and quotes it faithfully.

### The harness immediately found a bug, and groundedness could not see it

The corpus carries the same runbook per service and region on purpose, so the top chunks are
often near-duplicates. Sentence selection had no dedup, and answers were spending all three
slots restating one sentence three times. Adding dedup moved:

| | before | after |
|---|---|---|
| fact_coverage | 0.3222 | **0.3889** (+21%) |
| attribution | 0.5333 | 0.5500 |
| **groundedness** | **0.9593** | **0.9593** (unchanged) |

Two thirds of the answer was duplicate text and the groundedness score did not move by a
single digit — copied text is traceable text, whether or not it is copied three times. The
metric that would be quoted as evidence of answer quality was structurally incapable of
seeing the defect. That is the argument for the other four columns in one line.

### Knowing when not to answer

Three abstention signals were implemented and benchmarked rather than blended. Separation
between answerable and held-out unanswerable queries, as AUC:

| signal | bm25 | dense |
|---|---|---|
| **top_score** — the raw top retrieval score | **0.6937** | **0.7722** |
| coverage — query-term overlap with the best sentence | 0.6292 | 0.6111 |
| margin — how far the top result stands out | 0.5549 | 0.6465 |

The crudest signal won and the two designed ones are near-random. `coverage` has a structural
defect worth naming: this corpus contains queries written to share *no content word* with
their own relevant document — the queries dense retrieval exists to win — so a lexical
confidence signal scores them near zero whether or not an answer exists.

Thresholds are fitted at the 5th percentile of the answerable distribution, **using no
unanswerable examples at all**, so those 12 queries remain a genuine test. What that choice
buys, and what a different one would cost:

| answerable lost | unanswerable caught (bm25) | unanswerable caught (dense) |
|---|---|---|
| 5% | 0.3333 | 0.2500 |
| 20% | 0.3333 | 0.4167 |
| 30% | 0.3333 | 0.5833 |
| 40% | 0.5000 | 0.8333 |

BM25's curve is flat: six times the cost for nothing. Dense buys real ground. That difference
is invisible in the AUCs (0.69 vs 0.77) and is the reason the curve is published rather than a
single number — **better separation did not mean better behaviour at the operating point.**

### Why abstention is served for dense and not for BM25

Turning it on for BM25 declined `ERR-4021 remediation steps` — a query BM25 answers
**perfectly** (nDCG@10 1.0000 on exact-identifier queries), and the repository's own demo
query. The cause is IDF: that answer is duplicated across fifteen near-identical runbooks, so
the term is common and the score is low.

| answers appear in | mean BM25 top score |
|---|---|
| many documents (n=20) | **4.116** |
| a single document (n=40) | **11.332** |

The threshold lands on the exact-identifier block — the queries BM25 is best at. A raw BM25
score mixes "how well was this matched" with "how rare are these terms", and abstention needs
only the first; dense cosines are bounded and normalised, so they mean the same thing from one
query to the next. **A signal can rank well and still be the wrong thing to threshold on**, and
an AUC of 0.6937 said nothing about it. BM25's threshold is fitted, recorded, and deliberately
not served.

This is where the honest limit sits: at the served threshold the system still answers three
quarters of the questions it should refuse, and under the `bm25` default that
`docker-compose` uses for fast startup it does not abstain at all. Fixing it needs a signal
these lexical features do not contain. That is written up as a next step, not rounded away.

### The LLM path

The same harness scores the LLM path unchanged — it is one environment variable away and has
never been the source of a reported number, because a metric that depends on a vendor, a model
version and a sampling temperature is not reproducible by a reviewer. The abstention check runs
*before* the model call, so a question the corpus cannot answer is also the call not paid for.

The top retrieval score is reported as `retrieval_score`, not `confidence` — it says how
well the query matched, not how likely the answer is correct.

## API

- `GET /health` — status, active retriever, whether a reranker is fitted, LLM state
- `GET /metrics`
- `GET /events` protected when `API_KEY` is set
- `GET /query?q=...` and `POST /query`

```bash
curl "http://localhost:8000/query?q=ERR-4021%20remediation%20steps"
```

### Drift detection

A fitted model degrades quietly: nothing throws, the numbers simply stop
describing the world. This service watches its **top retrieval score** — how well the corpus answers the questions arriving — and reports
a status rather than raising an alarm.

```bash
curl localhost:8000/v1/drift
```

numeric PSI against the score distribution the evaluation query set produces on this corpus. Population Stability Index, read against the conventional thresholds:
below 0.10 stable, below 0.25 a moderate shift worth looking at, at or above 0.25
a significant one.

Three states exist besides a verdict, and each is reported rather than guessed:
`insufficient_data` below 50 observations, `no_reference` when training left none,
and a count of classes the reference never saw.

**Classifier confidence was tried first and rejected.** On a template-generated
corpus it is bimodal — near 1.0 on phrasings the model effectively memorised, much
lower on anything else — so PSI swung on ordinary traffic and reported drift that
was not there. A monitor that cries wolf teaches people to ignore it. The reasoning
is recorded in `monitoring/drift.py`.

Tests assert the monitor is **quiet on in-distribution data and loud on shifted
data**. One direction alone would not be evidence of anything.

### Distributed tracing

Every event is stored with the request id that produced it, and `/v1/events`
accepts a `request_id` filter. That is what makes a cross-service trace joinable:
the id already crossed service boundaries, but until it was recorded next to the
event there was nothing to join on.

```bash
curl "localhost:8000/v1/events?request_id=demo-1a2b3c4d"
```

The portfolio repo's `scripts/trace.py` asks all five services this question and
merges the answers into one ordered timeline.

## API versioning

Data endpoints are served under **`/v1`**. The same endpoints remain available at
the unversioned path as a **deprecated alias**, so consumers written before
versioning keep working; new callers should use `/v1`.

```bash
curl localhost:8000/version
```

`/health`, `/metrics` and `/version` are deliberately **not** versioned. They
describe the process rather than the API, and a monitoring system should not have
to follow an API version bump to keep scraping.

Both paths are served by one set of handlers, so the alias cannot drift from the
versioned route. `tests/test_api_versioning.py` asserts every data endpoint is
reachable under `/v1`, that the alias still exists, and that infrastructure
endpoints stay unversioned.

Why it matters here: without a version prefix there is no way for this service to
change a response shape without breaking every consumer on the same deploy. The
consumer-driven contract checks in the portfolio repo *detect* that breakage —
they do not prevent it.

## Run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q
python evaluation/harness.py
uvicorn api.server:app --reload --port 8000
```

**The service defaults to `RETRIEVER=bm25`** so a fresh clone starts instantly with no
model download. That is a startup-time choice, not a quality claim — dense scores
materially better. To use it:

```bash
RETRIEVER=dense uvicorn api.server:app --reload --port 8000
```

The embedding model (~440MB) downloads on first use and is cached thereafter.

Try the difference yourself — this query shares no content word with its target
document, so BM25 cannot find it and dense can:

```bash
curl "http://localhost:8000/query?q=customers%20cannot%20complete%20purchases%20at%20the%20final%20step"
```

Benchmarks:

```bash
python scripts/prefetch_beir.py --dataset nfcorpus
python evaluation/harness.py --beir nfcorpus
```

Reproducibility checks — this is what CI runs:

```bash
python training/generate_corpus.py --check
python training/train_reranker.py --verify --retriever bm25
```

## Cross-service integration (optional)

This service accepts documents so other services can contribute knowledge:

```
POST /documents   {"doc_id": ..., "title": ..., "text": ..., "source": ...}
```

The meeting intelligence service uses it to index decisions and action items, so
meeting outcomes become searchable alongside the policy corpus. The customer
operations service queries `/query` for a grounding passage when replying to a
customer.

Ingestion rebuilds the whole index per document. That is honest about what it is:
a demonstration of the integration, fine for hundreds of documents and wrong for
millions. Incremental indexing is not implemented.

Retrieval quality, the bench, and the committed metrics are unaffected — this adds
a write path, it does not change how ranking works.

## Reviewer Status

**What is real and independently checkable:**

- Four genuinely different retrievers behind one interface, compared by one harness.
- **Real BEIR benchmarks** with human relevance judgments, not only synthetic data —
  and the synthetic corpus's main findings replicate there.
- Raw numbers for both tracks in [`evaluation/results.json`](evaluation/results.json).
- Correct IR metrics — nDCG@k, Recall@k, MRR — replacing a `precision_at_k` that
  ignored `k` and substring-matched a joined string.
- A synthetic corpus engineered so the methods disagree, with the
  `vocabulary_mismatch` guarantee enforced by a test rather than asserted.
- A fitted reranker whose **null result is reported**, and a pretrained
  cross-encoder benched against it to prove the null was the task and not the model.
- Groundedness as a checkable property of each answer — **and four further metrics that
  show why groundedness alone certifies nothing**: fact coverage, attribution, context
  utilisation and hallucination rate on 12 held-out unanswerable questions.
- An abstention threshold calibrated **without ever looking at the unanswerable set**,
  with the operating curve published alongside the chosen point.
- [Architecture decision records](docs/adr/) for the five contested choices, each with the
  alternatives that were rejected and what would make it worth revisiting.
- Corpus, reranker, generation eval set and abstention threshold reproducible, verified
  in CI.

**What is explicitly not real:**

- The synthetic corpus is small (108 documents) and template-generated. BEIR carries
  the credible numbers; the synthetic corpus carries the offline demo and the
  per-query-type breakdown.
- Only NFCorpus was run from BEIR. `scifact` and `fiqa` are configured and one command
  away; FiQA is roughly a ten-hour encode on this CPU-only machine.
- Answers are extractive. There is no abstractive generation without an LLM configured,
  and no reported metric uses the LLM path.
- **Abstention is weak.** At the served threshold the system still answers three quarters
  of the questions it cannot answer, and it is switched off entirely under the `bm25`
  default because a raw BM25 score is not comparable across queries. A learned classifier
  over more features is the fix, and there are only 12 genuine negatives to fit it on.
- Answer quality is measured lexically. Fact coverage counts term groups, not meaning, so
  a correct paraphrase scores zero. That understates the generator and is stated rather
  than corrected by switching to a model-graded metric that could not be reproduced.
- No query rewriting, no HyDE, no multi-hop retrieval, no chunk-boundary tuning.
- No incremental indexing — the index is rebuilt from scratch on ingest.
- The SQLite event store is an audit trail for demos, not a production system.
- Docker/Compose/Kubernetes config is validated by static inspection and CI image
  builds; **cloud deployment is pending and unverified**.

**Shared scaffolding:** HTTP hardening (`utils/security.py`), the event store
(`utils/storage.py`) and the metrics surface follow a common service template shared
with the other Python services in this portfolio. That is deliberate reuse.

**Portfolio index:** https://github.com/Adityansh-Chand/ai-engineering-portfolio

## License

MIT
