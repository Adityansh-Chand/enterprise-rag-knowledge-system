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
