import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from rag import answer_question, build_index, reset_collection
from config import CHUNK_SIZE, OVERLAP, ENABLE_INGEST, RATE_LIMIT
from keyword_search import build_keyword_index
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1_000
MAX_DOCUMENT_CHARS = 200_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.collection = reset_collection()
    for path in sorted(Path("data").glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        app.state.collection = build_index(text, path.name, CHUNK_SIZE, OVERLAP)
    app.state.keyword_index = build_keyword_index(app.state.collection)
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

    try:
        result = answer_question(
            payload.question,
            app.state.collection,
            keyword_index=app.state.keyword_index,
        )
    except Exception:
        logger.exception("Answering question failed")
        raise HTTPException(
            status_code=502, detail="The language model request failed."
        )

    return QueryResponse(answer=result["answer"], sources=result["sources"])


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    if not ENABLE_INGEST:
        raise HTTPException(
            status_code=403, detail="Ingestion is disabled on this deployment"
        )

    try:
        app.state.collection = build_index(
            request.text, request.source, CHUNK_SIZE, OVERLAP
        )
    except Exception:
        logger.exception("Indexing document failed")
        raise HTTPException(status_code=502, detail="The embedding request failed.")
    app.state.keyword_index = build_keyword_index(app.state.collection)

    return {"status": "successful", "chunks": app.state.collection.count()}
