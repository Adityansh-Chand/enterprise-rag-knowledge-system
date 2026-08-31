import json
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import RETRIEVER, TOP_K
from llm import client as llm
from monitoring.drift import DriftMonitor
from monitoring.metrics import metrics
from rag.pipeline import RAGPipeline
from rag.reranker import is_fitted
from utils.security import (
    current_request_id,
    request_id_middleware,
    require_api_key,
)
from utils.storage import recent_events, save_event


app = FastAPI(title="Enterprise RAG Knowledge System", version="1.0.0")
app.middleware("http")(request_id_middleware)

API_VERSION = "v1"

# Data endpoints live on a router so they can be served at BOTH /v1/... and the
# original unversioned paths from a single definition. Without a version prefix
# there is no way to change a response shape without breaking every consumer on
# the same deploy -- the contract checks in the portfolio repo detect that
# breakage, they do not prevent it.
#
# The unversioned alias is kept because consumers already call it. It is the
# deprecation path, not a permanent second interface.
api = APIRouter()

pipeline = RAGPipeline()

# Top retrieval score against the distribution the evaluation query set produces
# on this corpus. A collapse means questions are arriving that the corpus no
# longer answers well -- worth knowing before someone reports bad answers.
# Both ends move here: the meeting service ingests into this corpus.
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"
drift_monitor = DriftMonitor.from_file(
    ARTIFACT_DIR / "drift_reference.json", name="retrieval_score"
)
CORPUS_PATH = Path(__file__).resolve().parents[1] / "datasets" / "corpus.json"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class DocumentRequest(BaseModel):
    """A passage to add to the searchable corpus."""

    doc_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=20000)
    title: str = Field("", max_length=500)
    source: str = Field("api", max_length=100)


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


@api.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20, request_id: str | None = None):
    """Recent events, optionally narrowed to one request id.

    `request_id` is what makes this endpoint a trace source rather than a log
    tail: the portfolio's scripts/trace.py asks all five services the same
    question and joins the answers into one timeline.
    """
    return {"events": recent_events(limit=min(limit, 100), request_id=request_id)}


@api.post("/documents", dependencies=[Depends(require_api_key)])
def ingest_document(request: DocumentRequest, http_request: Request):
    """Add a passage to the corpus and rebuild the index.

    This is what lets other services contribute knowledge -- the meeting service
    indexes the decisions and action items it extracts, so "what did we decide
    about the migration plan" becomes answerable from the same endpoint that
    answers policy questions.

    Rebuilding the whole index per document is honest about what this is: a
    demonstration of the integration, not an incremental-indexing implementation.
    Fine for hundreds of documents, wrong for millions.
    """
    metrics.increment("documents_ingested_total")
    chunks = pipeline.ingest_document(
        request.text, title=request.title, doc_id=request.doc_id
    )
    pipeline.build_index()
    save_event(
        "rag_ingest",
        {"doc_id": request.doc_id, "source": request.source, "chunks": chunks},
        current_request_id(http_request),
    )
    return {
        "doc_id": request.doc_id,
        "chunks_added": chunks,
        "documents_indexed": pipeline.document_count,
        "source": request.source,
    }


@api.get("/query")
def query(q: str, http_request: Request, _: None = Depends(require_api_key)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    metrics.increment("rag_queries_total")
    response = pipeline.query(q)
    drift_monitor.observe(response["retrieval_score"])
    save_event(
        "rag_query",
        {"query": q, "retrieval_score": response["retrieval_score"],
         "groundedness": response["groundedness"], "mode": response["mode"]},
        current_request_id(http_request),
    )
    return {"response": response}


@api.post("/query", dependencies=[Depends(require_api_key)])
def query_post(request: QueryRequest, http_request: Request):
    metrics.increment("rag_queries_total")
    response = pipeline.query(request.query)
    drift_monitor.observe(response["retrieval_score"])
    save_event(
        "rag_query",
        {"query": request.query, "retrieval_score": response["retrieval_score"],
         "groundedness": response["groundedness"], "mode": response["mode"]},
    )
    return {"response": response}


@api.get("/drift", dependencies=[Depends(require_api_key)])
def drift():
    """Is the corpus still answering the kind of questions being asked?"""
    return drift_monitor.report()


@app.get("/version")
def version():
    """What this service speaks, so a consumer can check rather than assume."""
    return {
        "service": "enterprise-rag-knowledge-system",
        "current": API_VERSION,
        "supported": [API_VERSION],
        "unversioned_alias": {
            "status": "deprecated",
            "note": ("the same endpoints are served without a /v1 prefix for "
                     "consumers that predate versioning; new callers should use "
                     f"/{API_VERSION}"),
        },
    }


# Mounted twice, one set of handlers. The alias is hidden from the schema so the
# generated docs show one interface rather than two identical ones.
app.include_router(api, prefix=f"/{API_VERSION}")
app.include_router(api, include_in_schema=False)
