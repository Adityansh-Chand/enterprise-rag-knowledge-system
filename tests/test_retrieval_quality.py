"""Quality gate for retrieval on the synthetic corpus.

Guards the properties the corpus was built to have. If a change makes
`vocabulary_mismatch` queries lexically solvable, or drops retrieval quality,
these fail.

Only BM25 and LSA run here: both are fast and need no model download, so the
suite stays runnable on a cold clone and in CI. Dense and hybrid numbers come
from `evaluation/harness.py`.
"""
import json
from pathlib import Path

import pytest

from training.generate_corpus import build, content_words, verify_vocabulary_mismatch
from rag.metrics import evaluate_run
from rag.retrievers import BM25Retriever, LSARetriever

ROOT = Path(__file__).resolve().parents[1]

MIN_BM25_NDCG = 0.55
MIN_BM25_EXACT_IDENTIFIER_NDCG = 0.90
# BM25 provably cannot answer these -- the queries share no content word with
# their targets. A high score here means the corpus guarantee broke.
MAX_BM25_MISMATCH_NDCG = 0.30


@pytest.fixture(scope="module")
def corpus_and_queries():
    corpus = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "datasets" / "queries.json").read_text(encoding="utf-8"))
    return corpus, queries


def _run(retriever, corpus, queries):
    retriever.index([f"{d['title']}. {d['text']}" for d in corpus])
    doc_ids = [d["id"] for d in corpus]
    run = {q["id"]: [doc_ids[i] for i, _ in retriever.search(q["text"], 10)] for q in queries}
    qrels = {q["id"]: {d: 1 for d in q["relevant"]} for q in queries}
    return run, qrels


def _ndcg_for_type(run, qrels, queries, query_type):
    ids = [q["id"] for q in queries if q["type"] == query_type]
    return evaluate_run({i: run[i] for i in ids}, {i: qrels[i] for i in ids})["ndcg@10"]


def test_committed_corpus_matches_the_generator():
    documents, queries = build()
    committed_docs = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    committed_queries = json.loads((ROOT / "datasets" / "queries.json").read_text(encoding="utf-8"))
    assert documents == committed_docs
    assert queries == committed_queries


def test_vocabulary_mismatch_queries_share_no_content_word(corpus_and_queries):
    """The corpus's central guarantee, re-checked as a test."""
    corpus, queries = corpus_and_queries
    assert verify_vocabulary_mismatch(corpus, queries) == []


def test_mismatch_queries_are_not_trivially_empty(corpus_and_queries):
    """Guards against satisfying the guarantee with contentless queries."""
    _, queries = corpus_and_queries
    for query in queries:
        if query["type"] == "vocabulary_mismatch":
            assert len(content_words(query["text"])) >= 3


def test_bm25_clears_the_overall_bar(corpus_and_queries):
    corpus, queries = corpus_and_queries
    run, qrels = _run(BM25Retriever(), corpus, queries)
    assert evaluate_run(run, qrels)["ndcg@10"] >= MIN_BM25_NDCG


def test_bm25_is_strong_on_exact_identifiers(corpus_and_queries):
    corpus, queries = corpus_and_queries
    run, qrels = _run(BM25Retriever(), corpus, queries)
    score = _ndcg_for_type(run, qrels, queries, "exact_identifier")
    assert score >= MIN_BM25_EXACT_IDENTIFIER_NDCG


def test_bm25_fails_on_vocabulary_mismatch(corpus_and_queries):
    """A *low* score is the correct result and the reason dense retrieval earns its place."""
    corpus, queries = corpus_and_queries
    run, qrels = _run(BM25Retriever(), corpus, queries)
    score = _ndcg_for_type(run, qrels, queries, "vocabulary_mismatch")
    assert score <= MAX_BM25_MISMATCH_NDCG


def test_lsa_indexes_and_ranks(corpus_and_queries):
    corpus, queries = corpus_and_queries
    run, qrels = _run(LSARetriever(), corpus, queries)
    assert evaluate_run(run, qrels)["ndcg@10"] > 0.4


# --- weighted fusion ----------------------------------------------------------

def test_fusion_weights_reduce_to_the_single_component_at_the_endpoints():
    """A semantic weight of 1.0 must be exactly dense; 0.0 exactly lexical.

    The tuned weight selected on both corpora is 1.0, so this equivalence is what
    makes "tuned fusion ties dense" a fact about the data rather than a bug.
    """
    from rag.retrievers import BM25Retriever, HybridRetriever, LSARetriever

    documents = [
        "Error ERR-4021 means the payment gateway declined the authorization.",
        "Refunds are processed within five business days of approval.",
        "Elevated p99 latency was observed on the checkout service.",
    ]
    lexical, semantic = BM25Retriever(), LSARetriever(n_components=2)

    semantic_only = HybridRetriever(lexical, semantic, lexical_weight=0.0, semantic_weight=1.0)
    lexical_only = HybridRetriever(lexical, semantic, lexical_weight=1.0, semantic_weight=0.0)
    for retriever in (semantic_only, lexical_only):
        retriever.index(documents)
    semantic.index(documents)
    lexical.index(documents)

    query = "why did the card payment fail"
    assert [i for i, _ in semantic_only.search(query, 3)] == [
        i for i, _ in semantic.search(query, 3)
    ]
    assert [i for i, _ in lexical_only.search(query, 3)] == [
        i for i, _ in lexical.search(query, 3)
    ]


def test_weighted_and_unweighted_fusion_are_named_distinctly():
    """Results tables must never conflate a tuned run with an unweighted one."""
    from rag.retrievers import BM25Retriever, HybridRetriever, LSARetriever

    lexical, semantic = BM25Retriever(), LSARetriever(n_components=2)
    assert HybridRetriever(lexical, semantic).name == "hybrid(bm25+lsa)"
    assert "hybrid_w" in HybridRetriever(
        lexical, semantic, lexical_weight=0.2, semantic_weight=0.8
    ).name


def test_committed_fusion_weights_were_selected_on_a_dev_split():
    """The selection must not have touched the queries used to report."""
    import json

    path = ROOT / "models" / "artifacts" / "fusion_weights.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "never tuned on the queries used to report" in payload["note"].lower()
    assert 0.0 <= payload["semantic_weight"] <= 1.0
    assert payload["lexical_weight"] == pytest.approx(1.0 - payload["semantic_weight"])
    # Tuning must at least not have made fusion worse than the unweighted baseline.
    half = payload["report_half"]
    assert half["hybrid_tuned"] >= half["hybrid_unweighted"]
