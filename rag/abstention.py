"""Deciding when not to answer.

Retrieval always returns its top k. Ask it something the corpus cannot answer
and it returns the k least-bad documents with no indication that they are wrong,
and an extractive generator quotes one of them perfectly groundedly. Abstention
is not a nicety here -- it is the only thing between the system and a confident
wrong answer, and groundedness cannot see the problem because the answer really
was copied out of a retrieved chunk.

Three candidate signals were implemented and benchmarked separately rather than
blended into one number nobody can interpret. Separation is measured as AUC
between the answerable and held-out unanswerable query sets:

                       BM25     dense
    top_score        0.6937    0.7722
    coverage         0.6292    0.6111
    margin           0.5549    0.6465

**top_score** -- the top retrieval score, unmodified. It wins, and it is the
crudest of the three. Kept as the served signal on that evidence.

**coverage** -- share of query terms the best retrieved sentence matches. It has
a structural problem worth stating: this corpus contains `vocabulary_mismatch`
queries written to share no content word with their own relevant document, which
are exactly the queries dense retrieval exists to win. A lexical confidence
signal scores them near zero whether or not an answer exists, so it partly
measures stopword overlap. Near-useless, as measured.

**margin** -- how far the top result stands out from the rest of the top k.
Scale-free and appealing in principle. Close to random in practice: on BM25 the
long left tail of zero-scoring answerable queries collapses it, and on dense the
normalised cosines are too tightly packed for the gap to mean anything.

Thresholds are fitted per retriever, on answerable queries only, by
`training/calibrate_abstention.py` -- the unanswerable set is never used to
choose an operating point, so it stays a genuine test.
"""
import json
from pathlib import Path

ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "artifacts" / "abstention.json"
)

DEFAULT_SIGNAL = "top_score"


def top_score_signal(query, chunks, scores=None):
    """The top retrieval score. Retriever-specific scale, so fitted per retriever."""
    return float(scores[0]) if scores else 0.0


def coverage_signal(query, chunks, scores=None):
    """Highest query-term coverage among the retrieved chunks."""
    # Imported here rather than at module scope: the generator calls into this
    # module to make the decision, so a top-level import would close the loop.
    from rag.generator import select_sentences

    selected = select_sentences(query, chunks)
    return max((coverage for _, _, coverage in selected), default=0.0)


def margin_signal(query, chunks, scores=None):
    """How far the top score stands out from the mean of the rest of the top k."""
    if not scores or len(scores) < 2:
        return 0.0
    top = float(scores[0])
    if top <= 0:
        return 0.0
    rest = [float(s) for s in scores[1:]]
    return max(0.0, (top - sum(rest) / len(rest)) / top)


SIGNALS = {
    "top_score": top_score_signal,
    "coverage": coverage_signal,
    "margin": margin_signal,
}


def load(retriever=None):
    """Return (signal_name, threshold) for a retriever, or (name, None) if unfitted.

    A missing artifact, or a retriever nothing was fitted for, means never
    abstain. That is the pre-calibration behaviour: a fresh clone answers rather
    than refusing everything because a file it has never heard of is absent.
    """
    if retriever is None:
        from config import RETRIEVER

        retriever = RETRIEVER
    try:
        artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_SIGNAL, None
    name = artifact.get("signal", DEFAULT_SIGNAL)
    fitted = artifact.get("retrievers", {}).get(retriever, {})
    # A threshold can be fitted and still not served. BM25 scores are not
    # comparable across queries -- a query whose answer is duplicated across
    # fifteen documents has low IDF and therefore a low score no matter how well
    # it is answered -- so its threshold lands on the exact-identifier queries
    # BM25 handles best. Fitted, recorded, and deliberately not acted on.
    if not fitted.get("abstain", True):
        return name, None
    return name, fitted.get("thresholds", {}).get(name)


def should_abstain(query, chunks, scores=None, signal=None, threshold=None,
                   retriever=None):
    """True when the system should decline rather than answer."""
    if signal is None or threshold is None:
        loaded_signal, loaded_threshold = load(retriever)
        signal = signal or loaded_signal
        threshold = loaded_threshold if threshold is None else threshold
    if threshold is None:
        return False
    return SIGNALS[signal](query, chunks, scores) < threshold
