"""Quality gate and behaviour tests for answer generation.

The gate exists to fail. Thresholds sit below the measured values with enough
room that normal variation does not flake, and close enough that a real
regression trips them. Two of them are deliberately *upper* bounds -- an
extractive generator that suddenly reports perfect fact coverage has almost
certainly started matching something it should not, and that is a bug that would
otherwise look like an improvement.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.abstention import (
    SIGNALS,
    load,
    margin_signal,
    should_abstain,
    top_score_signal,
)
from rag.generation_metrics import (
    attribution,
    context_utilisation,
    fact_coverage,
    groundedness,
    separation_auc,
    terms,
)
from rag.generator import NO_ANSWER, generate_answer

RESULTS_PATH = ROOT / "evaluation" / "generation_results.json"
EVAL_PATH = ROOT / "datasets" / "generation_eval.json"
ARTIFACT_PATH = ROOT / "models" / "artifacts" / "abstention.json"


@pytest.fixture(scope="module")
def results():
    if not RESULTS_PATH.exists():
        pytest.skip("run evaluation/generation_bench.py first")
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The metrics themselves
# --------------------------------------------------------------------------

def test_terms_normalises_trailing_punctuation():
    # The BM25 tokenizer keeps "receipt." as one token. Without normalisation
    # every sentence-final word would silently fail to match.
    assert "receipt" in terms("Submit an itemised receipt.")
    assert "retry.max_attempts" in terms("The retry.max_attempts setting")


def test_groundedness_is_one_for_copied_text():
    chunks = ["Refunds are processed within five business days."]
    assert groundedness(chunks[0], chunks) == 1.0


def test_groundedness_falls_for_invented_text():
    chunks = ["Refunds are processed within five business days."]
    assert groundedness("Bananas are delicious and unrelated", chunks) < 0.5


def test_fact_coverage_requires_every_term_in_a_unit():
    facts = [["renew", "certificate"], ["redistribute", "every", "replica"]]
    assert fact_coverage("Renew the certificate.", facts) == 0.5
    # Half a unit earns nothing: a partial remediation step is not a step.
    assert fact_coverage("Redistribute it.", facts) == 0.0
    assert fact_coverage(
        "Renew the certificate and redistribute it to every replica.", facts
    ) == 1.0


def test_attribution_counts_only_relevant_sources():
    assert attribution(["a", "b"], ["a"]) == 0.5
    assert attribution([], ["a"]) == 0.0


def test_context_utilisation_measures_chunks_actually_used():
    assert context_utilisation(["a"], ["a", "b", "c", "d"]) == 0.25
    assert context_utilisation([], ["a"]) == 0.0


def test_separation_auc_handles_ties_at_half_credit():
    # BM25 scores whole groups of queries at exactly zero. A tie-blind
    # implementation would report 1.0 here and flatter the signal.
    assert separation_auc([1.0, 1.0], [1.0, 1.0]) == 0.5
    assert separation_auc([2.0, 3.0], [0.0, 1.0]) == 1.0


# --------------------------------------------------------------------------
# Abstention behaviour
# --------------------------------------------------------------------------

def test_unfitted_retriever_answers_rather_than_refuses():
    # A retriever nothing was calibrated for -- or a clone with no artifact at
    # all -- must still answer. Refusing everything because a file is missing is
    # the worse failure, and it would look like a working abstention feature.
    assert load("no-such-retriever")[1] is None
    assert should_abstain(
        "anything", ["some chunk"], [1.0], retriever="no-such-retriever"
    ) is False


def test_abstains_below_threshold_and_answers_above():
    chunks = ["Error ERR-4021 is emitted when the processor declines."]
    assert should_abstain("q", chunks, [0.1], signal="top_score", threshold=1.0)
    assert not should_abstain("q", chunks, [5.0], signal="top_score", threshold=1.0)


def test_generate_answer_reports_abstention_as_its_own_mode():
    chunks = ["Something plausible but unrelated."]
    answer, mode = generate_answer(
        "unanswerable thing", chunks, use_llm=False, scores=[0.01],
        signal="top_score", threshold=1.0,
    )
    assert mode == "abstained"
    assert answer == NO_ANSWER


def test_answer_does_not_repeat_a_sentence_from_duplicate_chunks():
    # The corpus holds the same runbook per service and region, so the top
    # chunks are routinely near-duplicates. Without dedup the answer spent all
    # three slots on one sentence -- and groundedness scored that 0.9593, the
    # same as the fixed version, which is why the fix needed its own test.
    sentence = "Error ERR-4021 is emitted when the processor declines."
    chunks = [f"Service: {name}. {sentence}" for name in ("checkout", "billing", "search")]
    answer, mode = generate_answer(
        "ERR-4021 processor declines", chunks, use_llm=False, abstain=False
    )
    assert mode == "extractive"
    assert answer.count("Error ERR-4021 is emitted") == 1


def test_margin_signal_is_zero_without_a_comparison():
    assert margin_signal("q", [], [5.0]) == 0.0
    assert margin_signal("q", [], []) == 0.0


def test_top_score_signal_reads_the_first_score():
    assert top_score_signal("q", [], [3.5, 1.0]) == 3.5
    assert top_score_signal("q", [], []) == 0.0


# --------------------------------------------------------------------------
# Fitted artifacts and the dataset they were fitted on
# --------------------------------------------------------------------------

def test_thresholds_were_fitted_without_the_unanswerable_set():
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact["fitted_on"] == "answerable queries only"
    assert artifact["signal"] in SIGNALS
    for retriever, fitted in artifact["retrievers"].items():
        assert artifact["signal"] in fitted["thresholds"], retriever


def test_bm25_abstention_is_fitted_but_not_served():
    # BM25 scores are not comparable across queries: an answer duplicated across
    # fifteen runbooks has low IDF and therefore a low score however well it is
    # answered, so the threshold lands on the exact-identifier queries BM25
    # handles best. Recorded, not acted on. See ADR-005.
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    bm25 = artifact["retrievers"]["bm25"]
    assert bm25["abstain"] is False
    assert bm25["thresholds"]["top_score"] > 0  # fitted, just not served
    assert load("bm25")[1] is None
    assert artifact["retrievers"]["dense"]["abstain"] is True
    assert load("dense")[1] is not None


def test_unanswerable_queries_really_are_unanswerable():
    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    corpus = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    corpus_terms = set()
    for document in corpus:
        corpus_terms |= terms(f"{document['title']} {document['text']}")
    for query in payload["unanswerable"]:
        assert query["absent"] not in corpus_terms, query["text"]


def test_every_gold_fact_is_reachable():
    # A fact whose terms appear in no document would depress coverage forever
    # for a reason that has nothing to do with the generator.
    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    corpus = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    corpus_terms = set()
    for document in corpus:
        corpus_terms |= terms(f"{document['title']} {document['text']}")
    for query in payload["answerable"]:
        for fact in query["facts"]:
            missing = [term for term in fact if term not in corpus_terms]
            assert not missing, f"{query['id']} fact {fact} unreachable: {missing}"


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def test_extractive_answers_stay_grounded(results):
    # Lower bound. Extraction copies sentences, so anything much below this
    # means the answer stopped coming from the retrieved context.
    assert results["bm25"]["answerable"]["groundedness"] >= 0.90


def test_groundedness_is_not_mistaken_for_correctness(results):
    # The point of the whole harness, asserted so it cannot quietly stop being
    # true: near-perfect groundedness sits next to a hallucination rate that
    # would be unacceptable in any real deployment.
    bm25 = results["bm25"]
    assert bm25["answerable"]["groundedness"] >= 0.90
    assert bm25["abstention"]["hallucination_rate"] >= 0.25


def test_fact_coverage_has_not_silently_become_perfect(results):
    # Upper bound. An extractive selector picking three sentences by query-term
    # overlap cannot convey every gold unit; if it reports that it does, the
    # matcher has broken, not the generator improved.
    coverage = results["bm25"]["answerable"]["fact_coverage"]
    assert 0.20 <= coverage <= 0.75


def test_attribution_stays_above_chance(results):
    assert results["bm25"]["answerable"]["attribution"] >= 0.40


def test_calibration_holds_the_operating_point_it_promised(results):
    # Fitted at the 5th percentile, so roughly 5% of answerable queries should
    # be declined. Much more means the threshold drifted off its own definition.
    for retriever, result in results.items():
        assert result["abstention"]["answerable_declined"] <= 0.15, retriever


def test_top_score_beats_the_signals_it_replaced(results):
    # The served signal was chosen on this evidence. If a rewrite makes one of
    # the rejected signals better, the choice needs revisiting rather than
    # silently standing.
    auc = results["dense"]["abstention"]["signal_auc"]
    assert auc["top_score"] > auc["coverage"]
    assert auc["top_score"] > auc["margin"]
