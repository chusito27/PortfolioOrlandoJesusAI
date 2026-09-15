"""Thin client over Ollama.

Deliberately narrow: embed() and chat(). Swapping this file for an OpenAI or
Azure OpenAI implementation is the only change needed to move off local models,
which is why nothing above this layer knows the provider exists.
"""
import asyncio

import httpx

from app.config import settings

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


async def embed(texts: list[str], *, batch: int = 32) -> list[list[float]]:
    """Embed a list of texts, preserving order."""
    out: list[list[float]] = []
    async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=_TIMEOUT) as client:
        for i in range(0, len(texts), batch):
            window = texts[i : i + batch]
            resp = await client.post(
                "/api/embed", json={"model": settings.embed_model, "input": window}
            )
            resp.raise_for_status()
            out.extend(resp.json()["embeddings"])
    return out


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]


async def chat(
    prompt: str, *, system: str | None = None, temperature: float = 0.0
) -> str:
    """Single-turn completion. temperature=0 so evaluation runs are repeatable."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=_TIMEOUT) as client:
        resp = await client.post(
            "/api/chat",
            json={
                "model": settings.chat_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def embed_sync(texts: list[str]) -> list[list[float]]:
    return asyncio.run(embed(texts))
