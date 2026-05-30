from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.metrics import metrics
from rag.pipeline import RAGPipeline
from utils.security import request_id_middleware, require_api_key
from utils.storage import recent_events, save_event


app = FastAPI(title="Enterprise RAG Knowledge System", version="1.0.0")
app.middleware("http")(request_id_middleware)

pipeline = RAGPipeline()
DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "knowledge_base" / "hr_policy.txt"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    metrics.increment("http_errors_total")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": str(request.url.path)},
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
    if pipeline.document_count == 0:
        pipeline.ingest_document(DATA_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "running", "documents": pipeline.document_count}


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
    save_event("rag_query", {"query": q, "confidence": response["confidence"]})
    return {"response": response}


@app.post("/query", dependencies=[Depends(require_api_key)])
def query_post(request: QueryRequest):
    metrics.increment("rag_queries_total")
    response = pipeline.query(request.query)
    save_event("rag_query", {"query": request.query, "confidence": response["confidence"]})
    return {"response": response}
