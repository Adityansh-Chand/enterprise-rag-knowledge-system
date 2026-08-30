"""Reciprocal rank fusion over a lexical and a semantic retriever.

This is what makes "hybrid" an honest word here: two retrievers with genuinely
independent failure modes, combined. BM25 misses paraphrase; dense misses rare
exact identifiers. Fusing them recovers most of both.

RRF combines *ranks*, not scores, which matters because BM25 scores are unbounded
term-weight sums while cosine similarities sit in [-1, 1]. Averaging those
directly would let whichever retriever happens to have a larger numeric range
dominate for reasons unrelated to relevance.

**Weighting matters more than the original unweighted version assumed.** With
equal weights this retriever scored *below* its own dense component on both the
synthetic corpus and BEIR/NFCorpus: when one component is much stronger, giving
the weaker one equal say drags the fusion down. `lexical_weight` and
`semantic_weight` exist so that balance can be set from data rather than assumed
-- see `training/tune_fusion.py`, which selects them on a held-out *dev* split of
queries and never on the queries used to report.
"""
from typing import Sequence

from rag.retrievers.base import Retriever

# Standard RRF damping. Larger values flatten the contribution of top ranks.
RRF_K = 60


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        lexical: Retriever,
        semantic: Retriever,
        rrf_k: int = RRF_K,
        lexical_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ):
        self.lexical = lexical
        self.semantic = semantic
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

        if lexical_weight == semantic_weight:
            self.name = f"hybrid({lexical.name}+{semantic.name})"
        else:
            self.name = (
                f"hybrid_w({lexical.name}:{lexical_weight:g}"
                f"+{semantic.name}:{semantic_weight:g})"
            )

    def index(self, documents: Sequence[str]) -> None:
        self.lexical.index(documents)
        self.semantic.index(documents)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        # Fuse over a deeper pool than requested; a document ranked 30th by one
        # retriever and 2nd by the other should still surface.
        depth = max(k * 4, 50)
        fused: dict[int, float] = {}
        for retriever, weight in (
            (self.lexical, self.lexical_weight),
            (self.semantic, self.semantic_weight),
        ):
            if weight == 0.0:
                continue
            for rank, (index, _) in enumerate(retriever.search(query, depth)):
                fused[index] = fused.get(index, 0.0) + weight / (self.rrf_k + rank + 1)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [(int(i), float(s)) for i, s in ranked[:k]]
