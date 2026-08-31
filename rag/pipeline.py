"""Ingest, retrieve, rerank, answer.

The retriever is chosen by configuration and satisfies one interface, so
swapping BM25 for a dense or hybrid retriever changes nothing downstream.
"""
from config import EMBEDDING_MODEL, RERANK_DEPTH, RETRIEVER, TOP_K
from rag.chunker import chunk_document
from rag.generator import build_response
from rag.reranker import is_fitted, rerank
from rag.retrievers import build_retriever


class RAGPipeline:
    def __init__(self, retriever_name=None, model_name=None, top_k=TOP_K,
                 cache_path=None):
        self.retriever_name = retriever_name or RETRIEVER
        self.model_name = model_name or EMBEDDING_MODEL
        self.top_k = top_k

        kwargs = {}
        if self.retriever_name in ("dense", "hybrid", "router"):
            kwargs["model_name"] = self.model_name
            if cache_path is not None:
                kwargs["cache_path"] = cache_path
        self.retriever = build_retriever(self.retriever_name, **kwargs)

        self._chunks = []
        self._titles = []
        self._doc_ids = []
        self._indexed = False
        self.document_count = 0

    def ingest_passage(self, text, title="", doc_id=None):
        """Add one already-passage-sized unit without re-chunking.

        Used when the source is a prepared corpus. Keeping the text identical to
        what the evaluation harness indexes means a cached embedding matrix
        stays valid for the service too.
        """
        self._chunks.append(text)
        self._titles.append(title)
        self._doc_ids.append(doc_id or f"doc{len(self._doc_ids)}")
        self.document_count += 1
        self._indexed = False

    def ingest_document(self, text, title="", doc_id=None):
        chunks = chunk_document(text)
        for position, chunk in enumerate(chunks):
            self._chunks.append(chunk)
            self._titles.append(title)
            self._doc_ids.append(
                f"{doc_id}#{position}" if doc_id else f"doc{self.document_count}#{position}"
            )
        self.document_count += 1
        self._indexed = False
        return len(chunks)

    def has_document(self, doc_id):
        return any(existing.startswith(f"{doc_id}#") or existing == doc_id
                   for existing in self._doc_ids)

    def build_index(self):
        if not self._chunks:
            raise RuntimeError("ingest at least one document before indexing")
        self.retriever.index(self._chunks)
        self._indexed = True

    def retrieve(self, text):
        """The final ranked context an answer would be built from.

        Separate from `query` so evaluation and threshold calibration can look
        at exactly the chunks the generator sees, without reaching into private
        state or re-implementing the retrieve-then-rerank order and drifting
        from it.
        """
        if not self._chunks:
            return []
        if not self._indexed:
            self.build_index()

        hits = self.retriever.search(text, max(RERANK_DEPTH, self.top_k))
        candidates = [
            (score, self._chunks[i], self._titles[i], self._doc_ids[i])
            for i, score in hits
        ]
        return rerank(text, candidates)[: self.top_k]

    def query(self, text):
        if not self._chunks:
            return build_response(text, [], retriever=self.retriever_name)
        reranked = self.retrieve(text)
        response = build_response(text, reranked, retriever=self.retriever_name)
        response["retriever"] = self.retriever.name
        response["reranked"] = is_fitted()
        return response
