from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import RAGPipeline


app = FastAPI()

pipeline = RAGPipeline()
DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "knowledge_base" / "hr_policy.txt"


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def load_docs():
    if pipeline.document_count == 0:
        pipeline.ingest_document(DATA_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "running", "documents": pipeline.document_count}


@app.get("/query")
def query(q:str):
    return {
        "response": pipeline.query(q)
    }


@app.post("/query")
def query_post(request: QueryRequest):
    return {"response": pipeline.query(request.query)}
