from rag.chunker import chunk_document
from rag.embedder import embed
from rag.retriever import VectorIndex
from rag.generator import build_response
from config import TOP_K


class RAGPipeline:

    def __init__(self):

        self.index = VectorIndex()

    def ingest_document(self, text):

        chunks = chunk_document(text)

        for c in chunks:

            self.index.add(c, embed(c))

    def query(self, q):

        q_vec = embed(q)

        results = self.index.search(q_vec, top_k=TOP_K)

        context = "\n".join([r[1] for r in results])

        return build_response(context, results, q)
