from rag.retrievers.base import Retriever
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.dense import DenseRetriever
from rag.retrievers.hybrid import HybridRetriever
from rag.retrievers.router import RouterRetriever
from rag.retrievers.lsa import LSARetriever

__all__ = [
    "Retriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "LSARetriever",
    "build_retriever",
]


def build_retriever(name: str, **kwargs):
    """Construct a retriever by name. Unknown names fail loudly with the options.

    The previous `EMBEDDING_PROVIDER` setting accepted one value and raised on
    everything else, which made a stub look like a plug-in point. This resolves
    to four genuinely different implementations.
    """
    name = name.lower()
    if name == "bm25":
        return BM25Retriever()
    if name == "lsa":
        return LSARetriever(**kwargs)
    if name == "dense":
        return DenseRetriever(**kwargs)
    if name == "hybrid":
        return HybridRetriever(BM25Retriever(), DenseRetriever(**kwargs))
    if name == "router":
        return RouterRetriever(BM25Retriever(), DenseRetriever(**kwargs))
    raise ValueError(
        f"unknown retriever {name!r}; supported: bm25, lsa, dense, hybrid, router"
    )
