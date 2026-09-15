"""Ingestion pipeline: markdown files in, embedded chunks out."""
import argparse
import asyncio
import pathlib
import sys

import frontmatter
from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.chunking import chunk_markdown  # noqa: E402
from app.db import connection, init_schema, reset  # noqa: E402
from app.llm import embed  # noqa: E402

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
DOC_URL = "https://learn.microsoft.com/en-us/dotnet/{slug}"
console = Console()


def _load(path: pathlib.Path) -> tuple[str, str, str]:
    post = frontmatter.load(path)
    slug = path.relative_to(RAW).with_suffix("").as_posix()
    title = str(post.get("title") or slug.rsplit("/", 1)[-1].replace("-", " ").title())
    return title, DOC_URL.format(slug=slug), post.content


async def main(do_reset: bool) -> None:
    files = sorted(RAW.rglob("*.md"))
    if not files:
        console.print("[red]No markdown found. Run scripts/fetch_docs.py first.[/red]")
        return

    init_schema()
    if do_reset:
        console.print("[yellow]Resetting the database...[/yellow]")
        reset()

    total_chunks = 0
    with Progress(console=console) as progress:
        task = progress.add_task("Ingesting", total=len(files))

        for path in files:
            title, url, body = _load(path)
            chunks = chunk_markdown(body)
            if not chunks:
                progress.advance(task)
                continue

            vectors = await embed([c.embedding_text() for c in chunks])

            with connection() as conn:
                doc_id = conn.execute(
                    """
                    INSERT INTO documents (source_path, title, url)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_path) DO UPDATE
                        SET title = EXCLUDED.title, url = EXCLUDED.url
                    RETURNING id
                    """,
                    (path.relative_to(RAW).as_posix(), title, url),
                ).fetchone()[0]

                conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO chunks
                            (document_id, ordinal, heading_path, content, n_chars, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (doc_id, c.ordinal, c.heading_path, c.content, c.n_chars, v)
                            for c, v in zip(chunks, vectors)
                        ],
                    )
                conn.commit()

            total_chunks += len(chunks)
            progress.advance(task)

    with connection() as conn:
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        stored = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    console.print(f"[green]Done.[/green] {docs} documents, {stored} chunks (this run: {total_chunks}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop and recreate the tables")
    asyncio.run(main(ap.parse_args().reset))
