"""Evaluation harness.

A RAG demo without one is a guess. This measures three things that actually
break in production:

  recall@k    Did retrieval put the right document in front of the model at all?
              Everything downstream is capped by this number.
  grounding   Did the answer cite passages, and were the cited ids real? An
              uncited answer is unverifiable; an invented id is a hallucination
              you can detect mechanically.
  refusal     On questions the corpus cannot answer, did it say so instead of
              inventing something? This is the metric that decides whether the
              system is safe to put in front of a user.

Run --ablate to compare retrieval strategies on the same questions. If hybrid
search and re-ranking do not move the numbers, they are not worth their cost.
"""
import argparse
import asyncio
import json
import pathlib
import sys
from dataclasses import dataclass, field

import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.generate import answer_question  # noqa: E402
from app.retrieval import search  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
console = Console()


@dataclass
class Case:
    question: str
    answerable: bool = True
    expect_path_contains: list[str] = field(default_factory=list)


@dataclass
class Result:
    case: Case
    retrieved_sources: list[str]
    hit: bool
    answered: bool
    n_citations: int
    answer: str


def load_cases(path: pathlib.Path) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Case(**c) for c in raw["cases"]]


def _matches(sources: list[str], expected: list[str]) -> bool:
    blob = " ".join(sources).lower()
    return any(e.lower() in blob for e in expected)


async def evaluate(cases: list[Case], *, mode: str, rerank: bool, k: int) -> list[Result]:
    results: list[Result] = []
    for case in cases:
        hits = await search(case.question, rerank=rerank, mode=mode, limit=k)
        sources = [f"{h.title} {h.heading_path}" for h in hits]

        hit = _matches(sources, case.expect_path_contains) if case.answerable else False

        answer = await answer_question(case.question, rerank=rerank)
        results.append(
            Result(
                case=case,
                retrieved_sources=sources,
                hit=hit,
                answered=answer.answered,
                n_citations=len(answer.cited_chunk_ids),
                answer=answer.text,
            )
        )
        console.print(f"  [dim]{'ok ' if hit or not case.answerable else 'MISS'}[/dim] {case.question[:70]}")
    return results


def summarise(results: list[Result]) -> dict:
    answerable = [r for r in results if r.case.answerable]
    unanswerable = [r for r in results if not r.case.answerable]

    recall = sum(r.hit for r in answerable) / len(answerable) if answerable else 0.0
    grounded = (
        sum(r.n_citations > 0 for r in answerable) / len(answerable) if answerable else 0.0
    )
    refusal = (
        sum(not r.answered for r in unanswerable) / len(unanswerable)
        if unanswerable
        else None
    )
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "recall_at_k": round(recall, 3),
        "grounding_rate": round(grounded, 3),
        "refusal_rate": round(refusal, 3) if refusal is not None else None,
    }


def _table(rows: list[tuple[str, dict]], k: int) -> Table:
    t = Table(title=f"Evaluation (k={k})", header_style="bold")
    t.add_column("Configuration")
    t.add_column(f"recall@{k}", justify="right")
    t.add_column("grounding", justify="right")
    t.add_column("refusal", justify="right")
    for name, m in rows:
        t.add_row(
            name,
            f"{m['recall_at_k']:.0%}",
            f"{m['grounding_rate']:.0%}",
            "n/a" if m["refusal_rate"] is None else f"{m['refusal_rate']:.0%}",
        )
    return t


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(HERE / "golden.yaml"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ablate", action="store_true", help="compare retrieval strategies")
    ap.add_argument("--report", default=str(HERE / "report.json"))
    args = ap.parse_args()

    cases = load_cases(pathlib.Path(args.golden))
    console.print(f"[bold]{len(cases)} cases[/bold] from {args.golden}\n")

    configs = (
        [
            ("vector only", "vector", False),
            ("keyword only", "keyword", False),
            ("hybrid (RRF)", "hybrid", False),
            ("hybrid + rerank", "hybrid", True),
        ]
        if args.ablate
        else [("hybrid + rerank", "hybrid", True)]
    )

    rows, report = [], {}
    for name, mode, rerank in configs:
        console.print(f"[bold cyan]{name}[/bold cyan]")
        results = await evaluate(cases, mode=mode, rerank=rerank, k=args.k)
        metrics = summarise(results)
        rows.append((name, metrics))
        report[name] = {
            "metrics": metrics,
            "cases": [
                {
                    "question": r.case.question,
                    "answerable": r.case.answerable,
                    "retrieval_hit": r.hit,
                    "answered": r.answered,
                    "citations": r.n_citations,
                }
                for r in results
            ],
        }
        console.print()

    console.print(_table(rows, args.k))
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"\n[dim]Full report written to {args.report}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
