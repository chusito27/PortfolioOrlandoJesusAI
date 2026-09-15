"""FastAPI surface. Two endpoints: ask (full answer) and search (retrieval only).

/search exists so retrieval can be evaluated and debugged without paying for
generation, which is how you find out whether a bad answer was the retriever's
fault or the model's.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.db import connection, init_schema
from app.generate import answer_question
from app.retrieval import search

app = FastAPI(
    title="dotnet-docs-rag",
    description="Hybrid RAG over the official .NET documentation, fully local.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    rerank: bool = True


@app.on_event("startup")
def _startup() -> None:
    init_schema()


@app.get("/health")
def health() -> dict:
    with connection() as conn:
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {
        "status": "ok",
        "documents": docs,
        "chunks": chunks,
        "embed_model": settings.embed_model,
        "chat_model": settings.chat_model,
    }


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(422, "question is empty")
    return (await answer_question(req.question, rerank=req.rerank)).to_dict()


@app.post("/search")
async def search_only(req: AskRequest) -> dict:
    hits = await search(req.question, rerank=req.rerank)
    return {
        "question": req.question,
        "hits": [
            {
                "id": h.chunk_id,
                "source": h.citation,
                "vector_rank": h.vector_rank,
                "keyword_rank": h.keyword_rank,
                "rrf_score": round(h.rrf_score, 5),
                "rerank_score": h.rerank_score,
                "preview": h.content[:300],
            }
            for h in hits
        ],
    }
