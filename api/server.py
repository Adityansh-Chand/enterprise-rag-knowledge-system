import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import RETRIEVER, TOP_K
from llm import client as llm
from monitoring.metrics import metrics
from rag.pipeline import RAGPipeline
from rag.reranker import is_fitted
from utils.security import request_id_middleware, require_api_key
from utils.storage import recent_events, save_event


app = FastAPI(title="Enterprise RAG Knowledge System", version="1.0.0")
app.middleware("http")(request_id_middleware)

pipeline = RAGPipeline()
CORPUS_PATH = Path(__file__).resolve().parents[1] / "datasets" / "corpus.json"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    metrics.increment("http_errors_total")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": str(request.url.path)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    metrics.increment("validation_errors_total")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request",
            "details": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    metrics.increment("unhandled_errors_total")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "path": str(request.url.path)},
    )


@app.on_event("startup")
def load_docs():
    """Index the synthetic corpus.

    Passages are indexed exactly as the evaluation harness indexes them, so the
    two are measuring the same thing.
    """
    if pipeline.document_count == 0:
        for document in json.loads(CORPUS_PATH.read_text(encoding="utf-8")):
            pipeline.ingest_passage(
                f"{document['title']}. {document['text']}",
                title=document["title"],
                doc_id=document["id"],
            )
        pipeline.build_index()


@app.get("/health")
def health():
    """Health plus what is actually serving, so it can be checked from outside."""
    return {
        "status": "running",
        "documents": pipeline.document_count,
        "retriever": RETRIEVER,
        "top_k": TOP_K,
        "reranker_fitted": is_fitted(),
        "llm": llm.status(),
        "corpus": "synthetic (datasets/generate_corpus.py) - not a benchmark",
    }


@app.get("/metrics")
def metrics_endpoint():
    return metrics.snapshot()


@app.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20):
    return {"events": recent_events(limit=min(limit, 100))}


@app.get("/query")
def query(q: str, _: None = Depends(require_api_key)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    metrics.increment("rag_queries_total")
    response = pipeline.query(q)
    save_event(
        "rag_query",
        {"query": q, "retrieval_score": response["retrieval_score"],
         "groundedness": response["groundedness"], "mode": response["mode"]},
    )
    return {"response": response}


@app.post("/query", dependencies=[Depends(require_api_key)])
def query_post(request: QueryRequest):
    metrics.increment("rag_queries_total")
    response = pipeline.query(request.query)
    save_event(
        "rag_query",
        {"query": request.query, "retrieval_score": response["retrieval_score"],
         "groundedness": response["groundedness"], "mode": response["mode"]},
    )
    return {"response": response}
