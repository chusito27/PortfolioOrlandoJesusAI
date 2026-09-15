"""Re-ranking strategies.

Retrieval optimises recall: get the answer somewhere in the top 30. Re-ranking
optimises precision: put it in the top 5 that reach the prompt. Two ways to do
the second step, with a real trade-off between them:

  llm             Ask the generation model to score each passage. No extra
                  dependency and no extra model to host, but it costs a full
                  LLM round trip on every query and its output has to be parsed
                  and defended against.

  cross-encoder   A small model trained for exactly this, scoring the (query,
                  passage) pair jointly rather than comparing two independent
                  embeddings. Runs locally in milliseconds, but adds PyTorch and
                  a model download to the deployment.

Both are measured in eval/sweep.py rather than argued about.
"""
import json
import re
from typing import Protocol

from app.config import settings
from app.llm import chat


class Hit(Protocol):
    chunk_id: int
    content: str
    rrf_score: float
    rerank_score: float | None

    @property
    def citation(self) -> str: ...


# --------------------------------------------------------------------------
# LLM re-ranking
# --------------------------------------------------------------------------

_SYSTEM = (
    "You score how useful a documentation passage is for answering a question. "
    "Reply with JSON only."
)

_PROMPT = """Question: {question}

Passages:
{passages}

For each passage id, score 0-10 how directly it helps answer the question.
10 means it contains the answer. 0 means it is unrelated.

Reply with JSON only, in this exact shape:
{{"scores": [{{"id": <id>, "score": <0-10>}}]}}"""


async def _rerank_llm(question: str, hits: list) -> list:
    passages = "\n\n".join(f"[{h.chunk_id}] {h.citation}\n{h.content[:700]}" for h in hits)
    raw = await chat(_PROMPT.format(question=question, passages=passages), system=_SYSTEM)

    scores: dict[int, float] = {}
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            for item in json.loads(match.group(0)).get("scores", []):
                scores[int(item["id"])] = float(item["score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A malformed re-rank must not take the request down. Falling back to
            # the fused order is a worse answer, not a failed one.
            return hits

    for hit in hits:
        hit.rerank_score = scores.get(hit.chunk_id)
    return hits


# --------------------------------------------------------------------------
# Cross-encoder re-ranking
# --------------------------------------------------------------------------

_cross_encoder = None


def _load_cross_encoder():
    """Load once, on first use.

    Deliberately lazy: importing sentence_transformers pulls in PyTorch, which
    costs seconds of startup that a deployment using the LLM re-ranker should
    not have to pay.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(settings.cross_encoder_model, max_length=512)
    return _cross_encoder


async def _rerank_cross_encoder(question: str, hits: list) -> list:
    model = _load_cross_encoder()
    pairs = [(question, f"{h.citation}\n{h.content[:1500]}") for h in hits]
    scores = model.predict(pairs)

    for hit, score in zip(hits, scores):
        hit.rerank_score = float(score)
    return hits


# --------------------------------------------------------------------------

STRATEGIES = {"llm": _rerank_llm, "cross-encoder": _rerank_cross_encoder}


async def rerank(question: str, hits: list, *, strategy: str | None = None) -> list:
    """Re-score and reorder hits. Falls back to the incoming order on failure."""
    if not hits:
        return hits

    name = strategy or settings.reranker
    if name == "none":
        return hits

    scored = await STRATEGIES[name](question, hits)

    # rrf_score breaks ties, so a reranker that scores several passages equally
    # still falls back to the retriever's own ordering rather than an arbitrary one.
    return sorted(
        scored,
        key=lambda h: (h.rerank_score if h.rerank_score is not None else -1e9, h.rrf_score),
        reverse=True,
    )
