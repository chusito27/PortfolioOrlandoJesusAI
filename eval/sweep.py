"""Retrieval-only ablation across k.

The full harness measures the whole pipeline, which means it pays for generation
on every case. This sweeps retrieval alone, so it runs in seconds and can look at
several values of k.

Why k matters: at k=5 a corpus this size leaves so much headroom that every
strategy saturates at 100% and the comparison says nothing. Tightening to k=1
is what exposes whether a strategy ranks the right passage *first*, which is the
property that actually decides what reaches the prompt.
"""
import asyncio
import pathlib
import sys

import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.retrieval import search  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
console = Console()

MODES = ("vector", "keyword", "hybrid")
KS = (1, 3, 5, 10)


def load_cases() -> list[dict]:
    raw = yaml.safe_load((HERE / "golden.yaml").read_text(encoding="utf-8"))
    return [c for c in raw["cases"] if c.get("answerable", True)]


def hit(sources: list[str], expected: list[str]) -> bool:
    blob = " ".join(sources).lower()
    return any(e.lower() in blob for e in expected)


async def main() -> None:
    cases = load_cases()
    console.print(f"[bold]{len(cases)} answerable cases[/bold], retrieval only\n")

    results: dict[str, dict[int, float]] = {}

    for mode in MODES:
        results[mode] = {}
        for k in KS:
            hits = 0
            for case in cases:
                got = await search(case["question"], rerank=False, mode=mode, limit=k)
                if hit([f"{h.title} {h.heading_path}" for h in got], case["expect_path_contains"]):
                    hits += 1
            results[mode][k] = hits / len(cases)
        console.print(f"  {mode:10} done")

    # Re-ranking needs the LLM, so it only runs at the k values that matter.
    results["hybrid+rerank"] = {}
    for k in (1, 3):
        hits = 0
        for case in cases:
            got = await search(case["question"], rerank=True, mode="hybrid", limit=k)
            if hit([f"{h.title} {h.heading_path}" for h in got], case["expect_path_contains"]):
                hits += 1
        results["hybrid+rerank"][k] = hits / len(cases)
    console.print("  hybrid+rerank done\n")

    table = Table(title="Retrieval recall by strategy and k", header_style="bold")
    table.add_column("Strategy")
    for k in KS:
        table.add_column(f"recall@{k}", justify="right")

    for name, by_k in results.items():
        table.add_row(name, *[f"{by_k[k]:.0%}" if k in by_k else "-" for k in KS])

    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
