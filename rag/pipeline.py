from rag.chunker import chunk_document

from rag.embedder import simple_embedding

from rag.retriever import VectorIndex

from rag.generator import build_prompt


class RAGPipeline:

    def __init__(self):

        self.index = VectorIndex()

    def ingest_document(self,text):

        chunks = chunk_document(text)

        for c in chunks:

            v = simple_embedding(c)

            self.index.add(c,v)

    def query(self,q):

        q_vec = simple_embedding(q)

        results = self.index.search(q_vec)

        context = "\n".join([r[1] for r in results])

        return build_prompt(context,q)
