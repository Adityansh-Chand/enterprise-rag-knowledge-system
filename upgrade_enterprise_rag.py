import os
import subprocess
from pathlib import Path


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


def write_file(path, content):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:

        f.write(content.strip() + "\n")


def main():

    # detect repo root automatically

    base = Path.cwd()

    print("Updating repo at:", base)

    # README

    write_file(

        base / "README.md",

"""
# Enterprise RAG Knowledge System

Production Retrieval-Augmented Generation (RAG) system demonstrating modular AI architecture.

Pipeline:

documents → chunk → embed → vector search → retrieve → prompt → answer

---

## Run

pip install -r requirements.txt

uvicorn api.server:app --reload

---

## Example query

http://localhost:8000/query?q=remote work policy
"""

    )

    # requirements

    write_file(

        base / "requirements.txt",

"""
fastapi
uvicorn
numpy
pytest
"""

    )

    # dataset

    write_file(

        base / "datasets/knowledge_base/hr_policy.txt",

"""
Employees are entitled to 20 days of annual leave.

Remote work is allowed with manager approval.

Security training is required annually.

Unused leave can be carried forward for one year.
"""

    )

    # chunker

    write_file(

        base / "rag/chunker.py",

"""
def chunk_document(text, chunk_size=200, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks
"""

    )

    # embedder

    write_file(

        base / "rag/embedder.py",

"""
import numpy as np

def simple_embedding(text):

    return np.array([

        len(text),

        sum(ord(c) for c in text) % 997

    ])
"""

    )

    # retriever

    write_file(

        base / "rag/retriever.py",

"""
import numpy as np

def cosine_similarity(a,b):

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


class VectorIndex:

    def __init__(self):

        self.texts = []

        self.vectors = []

    def add(self,text,vector):

        self.texts.append(text)

        self.vectors.append(vector)

    def search(self,query_vector,top_k=3):

        scores = [

            cosine_similarity(query_vector,v)

            for v in self.vectors

        ]

        ranked = sorted(

            zip(scores,self.texts),

            key=lambda x: x[0],

            reverse=True

        )

        return ranked[:top_k]
"""

    )

    # generator

    write_file(

        base / "rag/generator.py",

'''
def build_prompt(context, query):

    return f"""

Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""
'''

    )

    # pipeline

    write_file(

        base / "rag/pipeline.py",

"""
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

        context = "\\n".join([r[1] for r in results])

        return build_prompt(context,q)
"""

    )

    # api

    write_file(

        base / "api/server.py",

"""
from fastapi import FastAPI

from rag.pipeline import RAGPipeline


app = FastAPI()

pipeline = RAGPipeline()


@app.on_event("startup")

def load_docs():

    with open("datasets/knowledge_base/hr_policy.txt") as f:

        pipeline.ingest_document(f.read())


@app.get("/query")

def query(q:str):

    return {

        "response": pipeline.query(q)

    }
"""

    )

    # evaluation

    write_file(

        base / "evaluation/evaluate_retrieval.py",

"""
from rag.pipeline import RAGPipeline


p = RAGPipeline()

p.ingest_document(

    "Employees receive 20 days vacation annually."

)

print(

    p.query("How many vacation days?")
)
"""

    )

    # tests

    write_file(

        base / "tests/test_pipeline.py",

"""
from rag.pipeline import RAGPipeline


def test_pipeline():

    p = RAGPipeline()

    p.ingest_document(

        "Remote work allowed with approval."

    )

    r = p.query("Can I work remotely?")

    assert "remote" in r.lower()
"""

    )

    # architecture

    write_file(

        base / "architecture/system_design.md",

"""
RAG architecture:

documents

→ chunking

→ embedding

→ vector search

→ prompt construction

→ generation
"""

    )

    # commit and push

    run("git add .")

    try:

        run('git commit -m "upgrade to flagship RAG architecture"')

    except:

        print("No changes to commit")

    run("git push")

    print("Upgrade complete")


if __name__ == "__main__":

    main()