"""The reranking comparison, and the conclusion drawn from it.

The fitted reranker measured +0.0000 nDCG@10, which left an open question: is
reranking worthless on this corpus, or was that model too weak? A pretrained
cross-encoder answers it, and the answer is that reranking does not pay here.

These tests protect that conclusion. If a future change makes a reranker actually
win, they should fail and be rewritten deliberately -- not quietly satisfied.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import cross_encoder  # noqa: E402

RESULTS_PATH = ROOT / "models" / "artifacts" / "rerank_comparison.json"

# nDCG@10 movement below this is noise at this corpus size, not an improvement.
MEANINGFUL = 0.01


@pytest.fixture(scope="module")
def results():
    if not RESULTS_PATH.exists():
        pytest.skip("run evaluation/rerank_bench.py first")
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def test_all_three_arms_were_measured_on_real_judgments(results):
    """A skipped arm must not be mistaken for an arm that ran and tied."""
    arms = {arm["arm"] for arm in results["nfcorpus"]}
    assert "no reranking" in arms
    assert "fitted pairwise" in arms
    assert any(arm.startswith("cross-encoder") for arm in arms)


def test_no_reranker_meaningfully_beats_not_reranking(results):
    """The finding: reranking does not pay on this corpus.

    Asserted rather than left implicit. A silent reversal -- from a model change,
    a library upgrade, or a corpus edit -- should surface here and force the
    README's claim to be rewritten, rather than leaving a stale conclusion in
    place next to better numbers.
    """
    for arm in results["nfcorpus"]:
        assert arm["delta_ndcg@10"] < MEANINGFUL, (
            f"{arm['arm']} now improves nDCG@10 by {arm['delta_ndcg@10']}. That may "
            "be real progress, but the README says reranking does not pay here -- "
            "update the claim deliberately."
        )


def test_the_cross_encoder_is_far_more_expensive(results):
    """The cost is half the argument, so it is asserted rather than described."""
    by_arm = {arm["arm"]: arm for arm in results["nfcorpus"]}
    cross = next(v for k, v in by_arm.items() if k.startswith("cross-encoder"))
    fitted = by_arm["fitted pairwise"]
    assert cross["rerank_ms_per_query"] > 100 * fitted["rerank_ms_per_query"]


def test_reranking_cannot_change_recall(results):
    """A sanity check on the harness itself.

    Reordering a fixed candidate list cannot add documents to it, so Recall@10
    must be identical across arms. If it is not, the arms are not being compared
    on the same candidates and every other number here is meaningless.
    """
    recalls = {round(arm["recall@10"], 6) for arm in results["nfcorpus"]}
    assert len(recalls) == 1, f"arms saw different candidates: {recalls}"


def test_rerank_is_a_no_op_without_the_model(monkeypatch):
    """An unavailable optional dependency must not reorder anything."""
    monkeypatch.setattr(cross_encoder, "_load", lambda name: None)
    candidates = [(0.9, "a", "d1"), (0.5, "b", "d2")]
    assert cross_encoder.rerank("q", candidates) == candidates


def test_rerank_preserves_every_candidate():
    """Nothing may be dropped: the tail past `depth` keeps its position."""
    if not cross_encoder.available():
        pytest.skip("cross-encoder weights unavailable")
    candidates = [(1.0 - i / 100, f"document number {i}", f"d{i}") for i in range(15)]
    reordered = cross_encoder.rerank("a query about documents", candidates, depth=5)
    assert len(reordered) == len(candidates)
    assert {c[2] for c in reordered} == {c[2] for c in candidates}
    # Everything below the shortlist is untouched.
    assert reordered[5:] == candidates[5:]
