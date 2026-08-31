"""A pretrained cross-encoder reranker, as the strong reference point.

The fitted pairwise reranker in `rag/reranker.py` measured **+0.0000 nDCG@10** --
honest, published, and not much of a result. The obvious question a reviewer asks
next is whether reranking itself is worthless here, or whether that particular
model was too weak. Those are very different answers and only a stronger reranker
separates them.

A cross-encoder is the right instrument for the comparison because it is a
categorically different thing from both the retriever and the fitted reranker: it
reads the query and the document *together* through a transformer and scores the
pair directly, rather than comparing two independently-computed representations.
That is what lets it catch relevance that no bag of features can see, and it is
also why it is too slow to use as a retriever and only viable over a shortlist.

Model choice, by the same rule used elsewhere -- best freely available that stays
practical:

    cross-encoder/ms-marco-MiniLM-L-6-v2   Apache 2.0, ~80MB

It is the standard MS MARCO reranker, small enough to download in CI and to run
on CPU over a shortlist of ten. The L-12 variant scores slightly higher and is
roughly twice the cost; if the L-6 result is decisive, the larger one changes the
size of the win rather than the conclusion.

Optional by design. When `sentence-transformers` or the model is unavailable this
reports that plainly and the caller falls back, exactly as the fitted reranker
returns input order when it has no artifact.
"""
import os
from functools import lru_cache

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Reranking every candidate would cost more than the retrieval it is improving.
# A cross-encoder is only affordable over a shortlist.
DEFAULT_DEPTH = 10


def model_name():
    return os.getenv("CROSS_ENCODER_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=2)
def _load(name):
    """Load once per process. Returns None when unavailable, never raises.

    A missing optional dependency must not take down a service whose primary path
    does not need it.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    try:
        return CrossEncoder(name)
    except Exception:  # noqa: BLE001 - offline, missing weights, no disk space
        return None


def available():
    return _load(model_name()) is not None


def rerank(query, candidates, depth=DEFAULT_DEPTH):
    """Reorder the top `depth` candidates by cross-encoder score.

    Candidates follow the same convention as `rag/reranker.py` and the pipeline:
    a tuple whose first element is the retrieval score and whose second is the
    text, with any further fields carried through untouched. Sharing one shape
    is what lets the two rerankers be swapped in a benchmark without adapters.

    Everything below `depth` keeps its position: the shortlist is reordered, the
    tail is not, which is what makes a cross-encoder affordable at all.
    """
    model = _load(model_name())
    if model is None or not candidates:
        return list(candidates)

    head, tail = list(candidates[:depth]), list(candidates[depth:])
    scores = model.predict([(query, candidate[1]) for candidate in head])

    ordered = sorted(zip(scores, head), key=lambda pair: pair[0], reverse=True)
    return [candidate for _, candidate in ordered] + tail
