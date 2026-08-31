"""Per-query routing: does it pick correctly, and does the pick help?

Two failure modes matter more than the headline number:

- a rule that fires too often collapses into lexical-only retrieval
- a rule that routes on the corpus's query-type labels scores beautifully and
  proves nothing, because those labels are not available at query time

So the tests check what the rule fires on, not only what it scores.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.retrievers.router import RouterRetriever, looks_lexical  # noqa: E402

RESULTS_PATH = ROOT / "models" / "artifacts" / "router_comparison.json"


class _Stub:
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.indexed = False

    def index(self, documents):
        self.indexed = True

    def search(self, query, k):
        return self.result


def test_identifier_queries_route_lexical():
    for query in ("ERR-4021 remediation steps",
                  "what does retry.max_attempts default to",
                  "POST /v2/invoices returns 409"):
        assert looks_lexical(query), query


def test_prose_routes_semantic():
    for query in ("customers cannot complete purchases at the final step",
                  "the website feels extremely sluggish for shoppers today",
                  "what are our net payment terms for enterprise invoices"):
        assert not looks_lexical(query), query


def test_an_acronym_alone_is_not_enough():
    """The rule must not fire on prose that merely contains an acronym.

    "what does our DPA say about sub-processors" is answered by a document using
    the expansion, so BM25 loses it (0.5000 against dense's 1.0000). Firing on
    any uppercase run would have routed it the wrong way.
    """
    assert not looks_lexical("what does our DPA say about sub-processors")
    assert not looks_lexical("MTTR expectations when tail latency regresses")


def test_router_delegates_and_counts():
    lexical = _Stub("bm25", [(1, 0.9)])
    semantic = _Stub("dense", [(2, 0.8)])
    router = RouterRetriever(lexical, semantic)

    assert router.search("ERR-4021 remediation steps", 5) == [(1, 0.9)]
    assert router.search("why is checkout slow", 5) == [(2, 0.8)]
    assert router.routed == {"lexical": 1, "semantic": 1}


def test_both_retrievers_are_indexed():
    """Routing decides per query, so both must be ready. The cost is real."""
    lexical, semantic = _Stub("bm25", []), _Stub("dense", [])
    RouterRetriever(lexical, semantic).index(["a document"])
    assert lexical.indexed and semantic.indexed


@pytest.fixture(scope="module")
def results():
    if not RESULTS_PATH.exists():
        pytest.skip("run training/tune_router.py first")
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def test_routing_is_selective_not_wholesale(results):
    """If the rule fired on everything it would just be BM25 with extra steps."""
    routing = results["dev_routing"]
    lexical, semantic = routing.get("lexical", 0), routing.get("semantic", 0)
    assert lexical > 0, "the rule never fires, so routing is a no-op"
    assert lexical < semantic, "the rule fires on most queries; that is lexical-only"


def test_the_rule_only_fires_on_identifier_queries(results):
    """Precision of the routing decision, measured on the dev half.

    This is the test that would catch a rule tuned to score well by firing
    indiscriminately: every lexical routing must be on an exact_identifier query.
    """
    lexical_routings = {
        key for key, count in results["dev_routing_by_type"].items()
        if key.endswith(":lexical") and count > 0
    }
    assert lexical_routings == {"exact_identifier:lexical"}, lexical_routings


def test_routing_beats_choosing_one_retriever_globally(results):
    """The claim being made, on the report half only."""
    report = results["report"]
    assert report["router"]["ndcg@10"] > report["dense"]["ndcg@10"]
    assert report["router"]["ndcg@10"] > report["bm25"]["ndcg@10"]
    assert report["router"]["ndcg@10"] > report["hybrid"]["ndcg@10"]


def test_routing_recovers_the_lexical_advantage_where_it_exists(results):
    """The mechanism, not just the aggregate.

    On identifier queries BM25 beats dense. If routing works, the router should
    match BM25 there and match dense everywhere else -- which is a stronger claim
    than the overall number, and the reason to believe the gain is not noise.
    """
    by_type = results["report_by_query_type"]
    identifier = by_type["exact_identifier"]
    assert identifier["bm25"] > identifier["dense"]
    assert identifier["router"] == pytest.approx(identifier["bm25"], abs=1e-6)

    for query_type, scores in by_type.items():
        if query_type == "exact_identifier":
            continue
        assert scores["router"] == pytest.approx(scores["dense"], abs=1e-6), query_type
