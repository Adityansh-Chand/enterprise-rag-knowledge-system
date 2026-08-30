"""Retrieval, chunking, metrics and generation behaviour."""
import json
from pathlib import Path

import pytest

from rag.chunker import chunk_document
from rag.generator import generate_answer, groundedness, select_sentences
from rag.metrics import evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k
from rag.retrievers import BM25Retriever, LSARetriever, build_retriever
from rag.retrievers.bm25 import tokenize

ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS = [
    "Error ERR-4021 is emitted when the upstream processor declines an authorization request.",
    "Enterprise invoices carry net sixty payment terms from the invoice issue date.",
    "Clause 7.3 limits aggregate liability to the fees paid in the preceding twelve months.",
    "The retry.max_attempts setting bounds how many times a failed webhook delivery is retried.",
    "Travel expenses are reimbursed when submitted within sixty days with an itemised receipt.",
]


def test_chunker_preserves_sentence_text():
    chunks = chunk_document("Remote work is allowed. Security training is annual.")
    assert chunks == ["Remote work is allowed. Security training is annual."]


def test_tokenizer_keeps_identifiers_intact():
    """Splitting these would destroy exactly the signal BM25 is good at."""
    assert "err-4021" in tokenize("what causes ERR-4021")
    assert "retry.max_attempts" in tokenize("retry.max_attempts default")
    assert "/v2/invoices" in tokenize("POST /v2/invoices returns 409")


@pytest.mark.parametrize("name", ["bm25", "lsa"])
def test_retrievers_rank_the_right_document_first(name):
    retriever = build_retriever(name)
    retriever.index(DOCUMENTS)
    hits = retriever.search("net sixty payment terms on invoices", 3)
    assert hits[0][0] == 1


def test_retriever_factory_rejects_unknown_names():
    """The old EMBEDDING_PROVIDER accepted one value and raised on all others."""
    with pytest.raises(ValueError) as error:
        build_retriever("word2vec")
    assert "bm25" in str(error.value)


def test_search_before_index_is_an_error():
    with pytest.raises(RuntimeError):
        BM25Retriever().search("anything", 3)


def test_bm25_finds_exact_identifiers():
    retriever = BM25Retriever()
    retriever.index(DOCUMENTS)
    assert retriever.search("ERR-4021", 1)[0][0] == 0


def test_lsa_reports_explained_variance():
    retriever = LSARetriever(n_components=3)
    retriever.index(DOCUMENTS)
    assert 0.0 < retriever.explained_variance <= 1.0


def test_ranking_metrics_reward_rank_order():
    qrels = {"q1": {"d1": 1}}
    assert ndcg_at_k({"q1": ["d1", "d2"]}, qrels, 10) == 1.0
    assert ndcg_at_k({"q1": ["d2", "d1"]}, qrels, 10) < 1.0
    assert mrr_at_k({"q1": ["d2", "d1"]}, qrels, 10) == 0.5
    assert recall_at_k({"q1": ["d2"]}, qrels, 1) == 0.0


def test_ndcg_is_sensitive_to_position():
    """The metric it replaced ignored k entirely and could not tell these apart."""
    qrels = {"q1": {"d1": 1}}
    first = ndcg_at_k({"q1": ["d1", "x", "y"]}, qrels, 10)
    third = ndcg_at_k({"q1": ["x", "y", "d1"]}, qrels, 10)
    assert first > third > 0


def test_evaluate_run_reports_every_cutoff():
    result = evaluate_run({"q1": ["d1"]}, {"q1": {"d1": 1}})
    for key in ("recall@1", "ndcg@1", "recall@10", "ndcg@10", "mrr@10"):
        assert key in result


def test_generator_selects_query_relevant_sentences():
    chunks = [
        "Refunds are processed within five business days. "
        "Error ERR-4021 means the processor declined the authorization."
    ]
    selected = select_sentences("ERR-4021 processor declined", chunks)
    assert "ERR-4021" in selected[0][0]


def test_generator_reports_no_answer_without_context():
    answer, mode = generate_answer("anything", [])
    assert mode == "no_context"
    assert "No relevant information" in answer


def test_groundedness_detects_unsupported_text():
    chunks = ["Refunds are processed within five business days."]
    assert groundedness("Refunds are processed within five business days.", chunks) == 1.0
    assert groundedness("Bananas are delicious and unrelated", chunks) < 0.5


def test_corpus_and_queries_are_consistent():
    corpus = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "datasets" / "queries.json").read_text(encoding="utf-8"))
    known = {d["id"] for d in corpus}

    assert corpus and queries
    for query in queries:
        assert query["relevant"], f"{query['id']} has no relevant documents"
        assert set(query["relevant"]) <= known, f"{query['id']} references unknown docs"
