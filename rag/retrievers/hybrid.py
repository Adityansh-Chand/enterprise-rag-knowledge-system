"""Reciprocal rank fusion over a lexical and a semantic retriever.

This is what makes "hybrid" an honest word here: two retrievers with genuinely
independent failure modes, combined. BM25 misses paraphrase; dense misses rare
exact identifiers. Fusing them recovers most of both.

RRF combines *ranks*, not scores, which matters because BM25 scores are unbounded
term-weight sums while cosine similarities sit in [-1, 1]. Averaging those
directly would let whichever retriever happens to have a larger numeric range
dominate for reasons unrelated to relevance.
"""
from typing import Sequence

from rag.retrievers.base import Retriever

# Standard RRF damping. Larger values flatten the contribution of top ranks.
RRF_K = 60


class HybridRetriever:
    name = "hybrid"

    def __init__(self, lexical: Retriever, semantic: Retriever, rrf_k: int = RRF_K):
        self.lexical = lexical
        self.semantic = semantic
        self.rrf_k = rrf_k
        self.name = f"hybrid({lexical.name}+{semantic.name})"

    def index(self, documents: Sequence[str]) -> None:
        self.lexical.index(documents)
        self.semantic.index(documents)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        # Fuse over a deeper pool than requested; a document ranked 30th by one
        # retriever and 2nd by the other should still surface.
        depth = max(k * 4, 50)
        fused: dict[int, float] = {}
        for retriever in (self.lexical, self.semantic):
            for rank, (index, _) in enumerate(retriever.search(query, depth)):
                fused[index] = fused.get(index, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [(int(i), float(s)) for i, s in ranked[:k]]
