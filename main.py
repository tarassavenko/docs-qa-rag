from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_question, build_index
from config import CHUNK_SIZE, OVERLAP

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    score: float
    id: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


with open("data/space.txt", encoding="utf-8") as f:
    text = f.read()

chunks, embedded_chunks = build_index(text, CHUNK_SIZE, OVERLAP)


@app.get("/")
def root():
    return {"name": "docs-qa-rag", "docs": "/docs"}


@app.get("/health")
def check_health():
    return {"status": "ok"}


@app.get("/status")
def check_status():
    return {"indexed": bool(chunks), "chunks": len(chunks)}


@app.post("/ask", response_model=QueryResponse)
def rag_query(request: QueryRequest):
    result = answer_question(request.question, embedded_chunks, chunks)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
