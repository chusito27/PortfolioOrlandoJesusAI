"""Answer generation with enforced citations.

The single rule that makes a RAG system trustworthy: the model may only use the
passages it was given, and every claim must point at one. A system that cannot
say "I don't know" will invent an answer, and in documentation that is worse
than silence.
"""
import re
from dataclasses import dataclass

from app.llm import chat
from app.retrieval import Hit, search

SYSTEM = """You answer questions about .NET using only the numbered passages provided.

Rules:
- Use only information present in the passages. Never rely on prior knowledge.
- Cite the passage id inline for every claim, like [3].
- If the passages do not contain the answer, reply exactly:
  NOT_IN_CONTEXT
  followed by one sentence saying what is missing.
- Be concise. Prefer a short answer with a code example over prose."""

PROMPT = """Question: {question}

Passages:
{context}

Answer, citing passage ids inline."""

CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited_chunk_ids: list[int]
    answered: bool

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "answered": self.answered,
            "citations": [
                {
                    "id": h.chunk_id,
                    "source": h.citation,
                    "url": h.url,
                    "rerank_score": h.rerank_score,
                }
                for h in self.hits
                if h.chunk_id in self.cited_chunk_ids
            ],
            "retrieved": [
                {
                    "id": h.chunk_id,
                    "source": h.citation,
                    "vector_rank": h.vector_rank,
                    "keyword_rank": h.keyword_rank,
                    "rrf_score": round(h.rrf_score, 5),
                    "rerank_score": h.rerank_score,
                }
                for h in self.hits
            ],
        }


def _format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{h.chunk_id}] Source: {h.citation}\n{h.content}" for h in hits
    )


async def answer_question(question: str, *, rerank: bool = True) -> Answer:
    hits = await search(question, rerank=rerank)

    if not hits:
        return Answer(question, "NOT_IN_CONTEXT\nNothing was retrieved for this question.", [], [], False)

    text = await chat(
        PROMPT.format(question=question, context=_format_context(hits)),
        system=SYSTEM,
    )

    valid = {h.chunk_id for h in hits}
    # Only count citations that point at a passage we actually supplied. A model
    # inventing [42] is exactly the failure this check exists to catch.
    cited = [int(c) for c in dict.fromkeys(CITATION.findall(text)) if int(c) in valid]

    return Answer(
        question=question,
        text=text.strip(),
        hits=hits,
        cited_chunk_ids=cited,
        answered=not text.strip().startswith("NOT_IN_CONTEXT"),
    )
