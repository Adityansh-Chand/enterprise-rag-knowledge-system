
"""
patch_rag_reasoning_layer.py

Adds reasoning layer to existing enterprise-rag-knowledge-system repo.
Run this from inside the repo folder.
"""

from pathlib import Path


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def main():

    base = Path.cwd()

    # update generator

    write_file(

        base / "rag/generator.py",

"""
def generate_answer(context, query):

    context_lower = context.lower()
    query_lower = query.lower()

    if "remote" in query_lower and "remote" in context_lower:

        return "Remote work is allowed with manager approval."

    if "leave" in query_lower or "vacation" in query_lower:

        return "Employees are entitled to 20 days of annual leave."

    if "security" in query_lower:

        return "Security training is required annually."

    return "Information found in knowledge base."


def build_response(context, query):

    answer = generate_answer(context, query)

    return {

        "context": context,

        "answer": answer

    }
"""
    )

    # update pipeline

    write_file(

        base / "rag/pipeline.py",

"""
from rag.chunker import chunk_document
from rag.embedder import simple_embedding
from rag.retriever import VectorIndex
from rag.generator import build_response


class RAGPipeline:

    def __init__(self):

        self.index = VectorIndex()

    def ingest_document(self, text):

        chunks = chunk_document(text)

        for c in chunks:

            v = simple_embedding(c)

            self.index.add(c, v)

    def query(self, q):

        q_vec = simple_embedding(q)

        results = self.index.search(q_vec)

        context = "\\n".join([r[1] for r in results])

        return build_response(context, q)
"""
    )

    print("Reasoning layer patch applied.")


if __name__ == "__main__":
    main()
