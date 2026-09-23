import logging
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from rag import answer_question, build_index, reset_collection
from config import (
    CHUNK_SIZE,
    OVERLAP,
    ENABLE_INGEST,
    RATE_LIMIT,
    LOG_LEVEL,
    RETRIEVAL_MODE,
    TOP_K,
)
from keyword_search import build_keyword_index
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

MAX_QUESTION_CHARS = 1_000
MAX_DOCUMENT_CHARS = 200_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    paths = sorted(Path("data").glob("*.txt"))
    app.state.collection = reset_collection()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        app.state.collection = build_index(text, path.name, CHUNK_SIZE, OVERLAP)
    app.state.keyword_index = build_keyword_index(app.state.collection)
    elapsed = time.perf_counter() - start
    if not paths:
        logger.warning("There are 0 documents at /data,index is empty")
    logger.info(
        "Indexed %d documents (%d chunks) in %.2fs|mode=%s top_k=%d "
        "chunk_size=%d overlap=%d",
        len(paths),
        app.state.collection.count(),
        elapsed,
        RETRIEVAL_MODE,
        TOP_K,
        CHUNK_SIZE,
        OVERLAP,
    )
    yield


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Source(BaseModel):
    score: float
    id: str
    text: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    source: str


class IngestResponse(BaseModel):
    chunks: int
    status: str


@app.get("/")
def root():
    return {"name": "docs-qa-rag", "docs": "/docs"}


@app.get("/health")
def check_health():
    return {"status": "ok"}


@app.get("/status")
def check_status():
    return {
        "indexed": app.state.collection.count() > 0,
        "chunks": app.state.collection.count(),
    }


@app.post("/ask", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT)
def rag_query(request: Request, payload: QueryRequest):
    if app.state.collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No document has been indexed yet. POST a document to /ingest first.",
        )
    start = time.perf_counter()

    try:
        result = answer_question(
            payload.question,
            app.state.collection,
            keyword_index=app.state.keyword_index,
        )
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("Answering question failed after %.0f ms", elapsed_ms)
        raise HTTPException(
            status_code=502, detail="The language model request failed."
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    top_source = result["sources"][0]["id"] if result["sources"] else "None"
    logger.info(
        "/ask completed in %.0fms|mode=%s top_source=%s sources=%d question=%r length_of_question=%d",
        elapsed_ms,
        RETRIEVAL_MODE,
        top_source,
        len(result["sources"]),
        payload.question[:80],
        len(payload.question),
    )

    return QueryResponse(answer=result["answer"], sources=result["sources"])


@app.post("/ingest", response_model=IngestResponse)
def ingest(ingest_request: IngestRequest, request: Request):
    if not ENABLE_INGEST:
        logger.warning(
            "Rejected /ingest for %r from %s — ENABLE_INGEST is false",
            ingest_request.source,
            client_ip(request),
        )
        raise HTTPException(
            status_code=403, detail="Ingestion is disabled on this deployment"
        )

    try:
        app.state.collection = build_index(
            ingest_request.text, ingest_request.source, CHUNK_SIZE, OVERLAP
        )
    except Exception:
        logger.exception("Indexing document failed")
        raise HTTPException(status_code=502, detail="The embedding request failed.")
    app.state.keyword_index = build_keyword_index(app.state.collection)

    return {"status": "successful", "chunks": app.state.collection.count()}
