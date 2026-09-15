"""Hybrid retrieval: vector + full-text, fused with RRF, then LLM re-ranking.

Why hybrid. Pure vector search is strong on meaning and weak on exact tokens:
ask for "AddScoped" and it happily returns passages about service lifetimes in
general. Pure keyword search is the mirror image. Running both and fusing the
rankings recovers what either one alone would drop.
"""
import json
import re
from dataclasses import dataclass, field

from app.config import settings
from app.db import connection
from app.llm import chat, embed_one


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    title: str
    url: str | None
    heading_path: str
    content: str
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    debug: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        where = f"{self.title} > {self.heading_path}" if self.heading_path else self.title
        return where


_SELECT = """
    SELECT c.id, c.document_id, d.title, d.url, c.heading_path, c.content
"""


def _vector_search(conn, embedding: list[float], k: int) -> list[tuple]:
    return conn.execute(
        _SELECT
        + """
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (embedding, k),
    ).fetchall()


def _keyword_search(conn, query: str, k: int) -> list[tuple]:
    return conn.execute(
        _SELECT
        + """
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.tsv @@ websearch_to_tsquery('english', %s)
        ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s)) DESC
        LIMIT %s
        """,
        (query, query, k),
    ).fetchall()


def _to_hit(row: tuple) -> Hit:
    return Hit(
        chunk_id=row[0],
        document_id=row[1],
        title=row[2],
        url=row[3],
        heading_path=row[4],
        content=row[5],
    )


def _fuse(vector_rows: list[tuple], keyword_rows: list[tuple]) -> list[Hit]:
    """Reciprocal Rank Fusion.

    RRF combines rankings rather than scores, which is what makes it safe here:
    a cosine distance and a ts_rank_cd score live on incomparable scales, so
    averaging them directly would be meaningless.
    """
    hits: dict[int, Hit] = {}

    for rank, row in enumerate(vector_rows, start=1):
        hit = hits.setdefault(row[0], _to_hit(row))
        hit.vector_rank = rank

    for rank, row in enumerate(keyword_rows, start=1):
        hit = hits.setdefault(row[0], _to_hit(row))
        hit.keyword_rank = rank

    k = settings.rrf_k
    for hit in hits.values():
        score = 0.0
        if hit.vector_rank:
            score += settings.rrf_vector_weight / (k + hit.vector_rank)
        if hit.keyword_rank:
            score += settings.rrf_keyword_weight / (k + hit.keyword_rank)
        hit.rrf_score = score

    return sorted(hits.values(), key=lambda h: h.rrf_score, reverse=True)


_RERANK_SYSTEM = (
    "You score how useful a documentation passage is for answering a question. "
    "Reply with JSON only."
)

_RERANK_PROMPT = """Question: {question}

Passages:
{passages}

For each passage id, score 0-10 how directly it helps answer the question.
10 means it contains the answer. 0 means it is unrelated.

Reply with JSON only, in this exact shape:
{{"scores": [{{"id": <id>, "score": <0-10>}}]}}"""


async def _rerank(question: str, hits: list[Hit]) -> list[Hit]:
    """Second-pass relevance scoring by the LLM.

    Retrieval optimises for recall (get the answer somewhere in the top 30).
    Re-ranking optimises for precision (put it in the top 5 that reach the
    prompt). Doing both is what stops the generator from being handed plausible
    but off-target context.
    """
    if not hits:
        return hits

    passages = "\n\n".join(
        f"[{h.chunk_id}] {h.citation}\n{h.content[:700]}" for h in hits
    )
    raw = await chat(
        _RERANK_PROMPT.format(question=question, passages=passages),
        system=_RERANK_SYSTEM,
    )

    scores: dict[int, float] = {}
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            for item in json.loads(match.group(0)).get("scores", []):
                scores[int(item["id"])] = float(item["score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A malformed rerank must not take the request down. Falling back to
            # the RRF order is a worse answer, not a failed one.
            return hits

    for hit in hits:
        hit.rerank_score = scores.get(hit.chunk_id)

    return sorted(
        hits,
        key=lambda h: (h.rerank_score if h.rerank_score is not None else -1, h.rrf_score),
        reverse=True,
    )


async def search(
    question: str,
    *,
    rerank: bool = True,
    mode: str = "hybrid",
    limit: int | None = None,
) -> list[Hit]:
    """Retrieve context for a question.

    mode is here so the eval harness can ablate the pipeline: running the same
    golden set through "vector", "keyword" and "hybrid" is the only honest way
    to claim the hybrid step earns its complexity.
    """
    vector_rows: list[tuple] = []
    keyword_rows: list[tuple] = []

    with connection() as conn:
        if mode in ("vector", "hybrid"):
            embedding = await embed_one(question)
            vector_rows = _vector_search(conn, embedding, settings.vector_top_k)
        if mode in ("keyword", "hybrid"):
            keyword_rows = _keyword_search(conn, question, settings.keyword_top_k)

    fused = _fuse(vector_rows, keyword_rows)
    top = limit or settings.context_chunks

    if not rerank:
        return fused[:top]

    reranked = await _rerank(question, fused[: settings.rerank_top_n])
    return reranked[:top]
