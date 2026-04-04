from rag.chunker import chunk_document
from rag.embedder import embed
from rag.retriever import HybridRetriever
from rag.reranker import rerank
from rag.generator import build_response
from config import TOP_K


class RAGPipeline:

    def __init__(self):

        self.index = HybridRetriever()


    def ingest_document(self,text):

        chunks = chunk_document(text)

        for c in chunks:

            self.index.add(c,embed(c))


    def query(self,q):

        q_vec = embed(q)

        results = self.index.search(q,q_vec,TOP_K)

        results = rerank(results)

        context = "\n".join([r[1] for r in results])

        return build_response(context,results,q)
