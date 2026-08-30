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

## Answers and groundedness

`rag/generator.py` selects the sentences that actually respond to the query, cites the
chunks they came from, and reports **groundedness**: the share of the answer's content
terms traceable to retrieved text. That is a checkable lexical property, not a model's
opinion of its own output.

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
- A fitted reranker whose **null result is reported**.
- Groundedness as a checkable property of each answer.
- Corpus and reranker reproducible, verified in CI.

**What is explicitly not real:**

- The synthetic corpus is small (108 documents) and template-generated. BEIR carries
  the credible numbers; the synthetic corpus carries the offline demo and the
  per-query-type breakdown.
- No cross-encoder reranker. A pretrained one would likely beat the fitted model; it is
  not implemented here, so no claim is made about it.
- No per-query routing between lexical and dense retrieval, which is what the fusion
  result actually points at. A single global weight cannot exploit BM25 being better on
  identifier queries and worse everywhere else.
- Only NFCorpus was run from BEIR. `scifact` and `fiqa` are configured and one command
  away; FiQA is roughly a ten-hour encode on this CPU-only machine.
- Answers are extractive. There is no abstractive generation without an LLM configured,
  and no reported metric uses the LLM path.
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
