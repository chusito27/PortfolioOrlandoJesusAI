"""Download a slice of the official .NET documentation.

Source: github.com/dotnet/docs, published under CC BY 4.0, so this corpus is
redistributable and the project can be public.

The areas below are chosen so the corpus exercises both halves of the hybrid
retriever. Conceptual pages (LINQ, microservices, testing strategy) are where
semantic search wins; reference pages (C# keywords, CLI commands, config keys)
are full of exact tokens that embeddings blur and full-text search nails.
"""
import argparse
import asyncio
import itertools
import pathlib

import httpx

TREE_URL = "https://api.github.com/repos/dotnet/docs/git/trees/main?recursive=1"
RAW = "https://raw.githubusercontent.com/dotnet/docs/main/{path}"

AREAS = (
    "docs/csharp/language-reference/",
    "docs/csharp/programming-guide/",
    "docs/standard/linq/",
    "docs/core/testing/",
    "docs/core/extensions/",
    "docs/core/diagnostics/",
    "docs/core/tools/",
    "docs/architecture/microservices/",
)

# Auto-generated warning pages and include fragments: hundreds of near-identical
# stubs that would dominate the corpus and teach the retriever nothing.
EXCLUDE = ("/warnings/", "/trim-warnings/", "/includes/", "/snippets/", "/misc/")

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"


def _select(paths: list[str], limit: int) -> list[str]:
    """Round-robin across areas so no single area swamps the corpus."""
    buckets = {a: [p for p in paths if p.startswith(a)] for a in AREAS}
    interleaved = itertools.chain.from_iterable(
        itertools.zip_longest(*buckets.values())
    )
    return [p for p in interleaved if p][:limit]


async def _download(client: httpx.AsyncClient, path: str, sem: asyncio.Semaphore) -> bool:
    async with sem:
        try:
            resp = await client.get(RAW.format(path=path), timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError:
            return False
    if len(resp.text) < 900:  # stubs and redirect pages
        return False
    dest = OUT / path.replace("docs/", "", 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text, encoding="utf-8")
    return True


async def main(limit: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        print("Listing the dotnet/docs tree...")
        tree = (await client.get(TREE_URL, timeout=60.0)).json()["tree"]

        candidates = [
            n["path"]
            for n in tree
            if n["type"] == "blob"
            and n["path"].endswith(".md")
            and n["path"].startswith(AREAS)
            and not any(x in n["path"] for x in EXCLUDE)
            and not n["path"].endswith("index.md")
        ]
        paths = _select(candidates, limit)

        print(f"Downloading {len(paths)} markdown files across {len(AREAS)} areas...")
        sem = asyncio.Semaphore(16)
        results = await asyncio.gather(*(_download(client, p, sem) for p in paths))

    print(f"Saved {sum(results)} files to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=320)
    asyncio.run(main(ap.parse_args().limit))
