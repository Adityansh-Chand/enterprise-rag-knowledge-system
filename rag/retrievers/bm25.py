"""BM25 -- the lexical baseline.

Term frequency with saturation and document-length normalisation. Not semantic:
a query term must literally appear for a document to score. That limitation is
the point -- it is the control against which any semantic method has to justify
itself, and on exact-identifier queries (error codes, API paths, clause numbers)
it is very hard to beat.
"""
import re
from typing import Sequence

from rank_bm25 import BM25Okapi

# The optional leading slash keeps API paths whole ("/v2/invoices").
_TOKEN = re.compile(r"/?[a-zA-Z0-9][a-zA-Z0-9._/-]*")


def tokenize(text: str) -> list[str]:
    """Keep dotted/slashed/hyphenated identifiers intact.

    `ERR-4021`, `retry.max_attempts` and `/v2/invoices` survive as single tokens
    rather than being shattered, which is exactly where BM25 earns its keep.
    """
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25 = None

    def index(self, documents: Sequence[str]) -> None:
        self._bm25 = BM25Okapi([tokenize(d) for d in documents], k1=self.k1, b=self.b)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        if self._bm25 is None:
            raise RuntimeError("index() must be called before search()")
        scores = self._bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]
