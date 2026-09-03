import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from rag import answer_question, build_index, reset_collection
from config import CHUNK_SIZE, OVERLAP

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1_000
MAX_DOCUMENT_CHARS = 200_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.collection = reset_collection()
    for path in sorted(Path("data").glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        app.state.collection = build_index(text, path.name, CHUNK_SIZE, OVERLAP)
    yield


app = FastAPI(lifespan=lifespan)


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
        "indexed": bool(app.state.collection),
        "chunks": app.state.collection.count(),
    }


@app.post("/ask", response_model=QueryResponse)
def rag_query(request: QueryRequest):
    if app.state.collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No document has been indexed yet. POST a document to /ingest first.",
        )

    try:
        result = answer_question(request.question, app.state.collection)
    except Exception:
        logger.exception("Answering question failed")
        raise HTTPException(
            status_code=502, detail="The language model request failed."
        )

    return QueryResponse(answer=result["answer"], sources=result["sources"])


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    try:
        app.state.collection = build_index(
            request.text, request.source, CHUNK_SIZE, OVERLAP
        )
    except Exception:
        logger.exception("Indexing document failed")
        raise HTTPException(status_code=502, detail="The embedding request failed.")

    return {"status": "successful", "chunks": app.state.collection.count()}
