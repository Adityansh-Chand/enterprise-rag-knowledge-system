"""Reranking that adds signal the retriever did not already have.

The previous implementation rescored candidates using the same `tokenize()`
overlap the retriever had just used, so it could not meaningfully reorder
anything -- its ranking function was a monotone transform of the input ranking.

A reranker is only worth running if its features are *independent* of the
retrieval score. This one scores each (query, document) pair on lexical overlap,
identifier matching, coverage, length and original rank, with weights **fitted**
by logistic regression on the corpus relevance judgments.

Trained by `training/train_reranker.py`. If no fitted artifact is present,
`rerank` returns the input order unchanged rather than silently applying an
invented heuristic -- an honest no-op beats a fake improvement.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "models" / "artifacts" / "reranker.json"

_TOKEN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*")
# Tokens that look like identifiers: error codes, config keys, API paths, versions.
_IDENTIFIER = re.compile(r"(?:[A-Z]{2,}-\d+|\w+\.\w+|/v\d+/\w+|\b\d+\.\d+\b)")

FEATURE_NAMES = [
    "term_overlap",
    "query_coverage",
    "identifier_match",
    "title_overlap",
    "length_ratio",
    "reciprocal_rank",
]


def _tokens(text):
    return {t.lower() for t in _TOKEN.findall(text)}


def features(query, document, rank, title=""):
    """Signals deliberately chosen to be computable without the retriever's score."""
    query_tokens = _tokens(query)
    doc_tokens = _tokens(document)
    if not query_tokens:
        return [0.0] * len(FEATURE_NAMES)

    overlap = query_tokens & doc_tokens
    query_ids = set(_IDENTIFIER.findall(query))
    doc_ids = set(_IDENTIFIER.findall(document))

    return [
        len(overlap) / len(query_tokens | doc_tokens),          # Jaccard
        len(overlap) / len(query_tokens),                        # coverage
        1.0 if (query_ids and query_ids & doc_ids) else 0.0,     # exact identifier hit
        len(query_tokens & _tokens(title)) / len(query_tokens),  # title overlap
        min(len(doc_tokens) / 200.0, 1.0),                       # length, saturating
        1.0 / (rank + 1),                                        # prior from retrieval
    ]


@lru_cache(maxsize=1)
def _weights():
    if not ARTIFACT_PATH.exists():
        return None
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    return payload["intercept"], [payload["weights"][name] for name in FEATURE_NAMES]


def is_fitted() -> bool:
    return _weights() is not None


def rerank(query, candidates):
    """Reorder (score, text) or (score, text, title) candidates.

    Returns the input order untouched when no fitted artifact exists.
    """
    fitted = _weights()
    if fitted is None or not candidates:
        return list(candidates)

    intercept, weights = fitted
    scored = []
    for rank, candidate in enumerate(candidates):
        text = candidate[1]
        title = candidate[2] if len(candidate) > 2 else ""
        vector = features(query, text, rank, title)
        logit = intercept + sum(w * v for w, v in zip(weights, vector))
        scored.append((logit, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored]
