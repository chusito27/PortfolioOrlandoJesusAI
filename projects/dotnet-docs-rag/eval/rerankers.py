"""Compare re-ranking strategies on recall and latency.

The question this answers: an LLM re-ranker needs no extra dependency but costs
a full model round trip per query, while a cross-encoder adds PyTorch and a model
download but runs locally in milliseconds. Which one actually retrieves better,
and what does the difference cost?

Latency here is the re-ranking step only, measured after retrieval, so the two
numbers are comparable.
"""
import asyncio
import pathlib
import statistics
import sys
import time

import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.rerank import rerank  # noqa: E402
from app.retrieval import search  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
console = Console()

STRATEGIES = ("none", "llm", "cross-encoder")
KS = (1, 3, 5)


def load_cases() -> list[dict]:
    raw = yaml.safe_load((HERE / "golden.yaml").read_text(encoding="utf-8"))
    return [c for c in raw["cases"] if c.get("answerable", True)]


def hit(sources: list[str], expected: list[str]) -> bool:
    blob = " ".join(sources).lower()
    return any(e.lower() in blob for e in expected)


async def main() -> None:
    cases = load_cases()
    console.print(f"[bold]{len(cases)} answerable cases[/bold], hybrid retrieval\n")

    # Retrieve once per question; every strategy re-ranks the same candidate set,
    # so the comparison isolates the re-ranker instead of the retriever.
    candidates = []
    for case in cases:
        fused = await search(case["question"], rerank=False, mode="hybrid",
                            limit=settings.rerank_top_n)
        candidates.append((case, fused))

    results: dict[str, dict] = {}

    for strategy in STRATEGIES:
        recalls = {k: 0 for k in KS}
        latencies: list[float] = []

        for case, fused in candidates:
            pool = [type(h)(**vars(h)) for h in fused]  # fresh copies per strategy
            t0 = time.perf_counter()
            ranked = await rerank(case["question"], pool, strategy=strategy)
            latencies.append((time.perf_counter() - t0) * 1000)

            for k in KS:
                if hit([f"{h.title} {h.heading_path}" for h in ranked[:k]],
                       case["expect_path_contains"]):
                    recalls[k] += 1

        results[strategy] = {
            "recall": {k: v / len(cases) for k, v in recalls.items()},
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        }
        console.print(f"  {strategy:14} done")

    table = Table(title="Re-ranking strategies (hybrid retrieval, same candidates)",
                  header_style="bold")
    table.add_column("Strategy")
    for k in KS:
        table.add_column(f"recall@{k}", justify="right")
    table.add_column("p50 latency", justify="right")
    table.add_column("p95 latency", justify="right")

    for name, r in results.items():
        table.add_row(
            name,
            *[f"{r['recall'][k]:.0%}" for k in KS],
            f"{r['p50']:.0f} ms",
            f"{r['p95']:.0f} ms",
        )

    console.print()
    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
