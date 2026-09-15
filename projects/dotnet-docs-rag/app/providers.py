"""Model providers.

Everything above this file asks for `embed()` and `chat()` and does not know who
answers. Two implementations ship:

  ollama    Local models. No API key, no per-query cost, nothing leaves the
            machine. The default, so the project is runnable by anyone who
            clones it.

  bedrock   AWS Bedrock via boto3, using Titan for embeddings and the Converse
            API for generation. The same pipeline against managed models, for a
            deployment that has to run in an AWS account.

Switching is a one-line change in .env. The one thing it is not free of:
embedding dimensions differ between providers (nomic-embed-text is 768, Titan v2
is 1024), and the chunks table declares a fixed vector width. Changing provider
means changing EMBED_DIM and re-ingesting, which is why the dimension is
configuration rather than a constant.
"""
import asyncio
import os
from typing import Protocol

import httpx

from app.config import settings

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class Provider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def chat(self, prompt: str, *, system: str | None, temperature: float) -> str: ...


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

class OllamaProvider:
    def __init__(self) -> None:
        self.base_url = settings.ollama_url
        self.embed_model = settings.embed_model
        self.chat_model = settings.chat_model

    async def embed(self, texts: list[str], *, batch: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        async with httpx.AsyncClient(base_url=self.base_url, timeout=_TIMEOUT) as client:
            for i in range(0, len(texts), batch):
                resp = await client.post(
                    "/api/embed",
                    json={"model": self.embed_model, "input": texts[i : i + batch]},
                )
                resp.raise_for_status()
                out.extend(resp.json()["embeddings"])
        return out

    async def chat(self, prompt: str, *, system: str | None = None,
                   temperature: float = 0.0) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        async with httpx.AsyncClient(base_url=self.base_url, timeout=_TIMEOUT) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]


# --------------------------------------------------------------------------
# AWS Bedrock
# --------------------------------------------------------------------------

class BedrockProvider:
    """AWS Bedrock through boto3.

    boto3 is synchronous, so every call is pushed to a worker thread rather than
    blocking the event loop. Embedding is issued as a batch of concurrent calls
    because Titan's InvokeModel takes one input text per request.
    """

    def __init__(self) -> None:
        import boto3  # imported here so the Ollama path needs no AWS dependency

        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self.embed_model = settings.bedrock_embed_model
        self.chat_model = settings.bedrock_chat_model
        self._sem = asyncio.Semaphore(8)

    async def _embed_one(self, text: str) -> list[float]:
        import json

        async with self._sem:
            resp = await asyncio.to_thread(
                self.client.invoke_model,
                modelId=self.embed_model,
                body=json.dumps({"inputText": text}),
            )
        return json.loads(resp["body"].read())["embedding"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return list(await asyncio.gather(*(self._embed_one(t) for t in texts)))

    async def chat(self, prompt: str, *, system: str | None = None,
                   temperature: float = 0.0) -> str:
        kwargs = {
            "modelId": self.chat_model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"temperature": temperature, "maxTokens": 2048},
        }
        if system:
            kwargs["system"] = [{"text": system}]

        resp = await asyncio.to_thread(self.client.converse, **kwargs)
        return resp["output"]["message"]["content"][0]["text"]


# --------------------------------------------------------------------------

_PROVIDERS = {"ollama": OllamaProvider, "bedrock": BedrockProvider}
_instance: Provider | None = None


def provider() -> Provider:
    global _instance
    if _instance is None:
        name = settings.provider
        if name not in _PROVIDERS:
            raise ValueError(f"unknown provider {name!r}, expected one of {list(_PROVIDERS)}")
        _instance = _PROVIDERS[name]()
    return _instance
