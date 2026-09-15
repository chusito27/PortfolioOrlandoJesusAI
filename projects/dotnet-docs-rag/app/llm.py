"""Model access for the rest of the application.

A thin facade over whichever provider is configured. Nothing above this file
knows whether the models are running locally or in AWS.
"""
import asyncio

from app.providers import provider


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, preserving order."""
    return await provider().embed(texts)


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]


async def chat(prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
    """Single-turn completion. temperature=0 so evaluation runs are repeatable."""
    return await provider().chat(prompt, system=system, temperature=temperature)


def embed_sync(texts: list[str]) -> list[list[float]]:
    return asyncio.run(embed(texts))
