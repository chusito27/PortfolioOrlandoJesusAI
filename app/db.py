"""Postgres access. One pool, one schema definition, nothing clever."""
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    source_path  TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    url          TEXT,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT NOT NULL,
    heading_path TEXT NOT NULL DEFAULT '',
    content      TEXT NOT NULL,
    n_chars      INT NOT NULL,
    embedding    vector(%(dim)s),
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, ordinal)
);

-- Semantic half of the hybrid search.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Lexical half. Exact tokens (API names, keywords) that embeddings are bad at.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON chunks USING gin (tsv);
"""

_pool: ConnectionPool | None = None


def _ensure_extension() -> None:
    """Create the vector extension once, on a standalone connection.

    register_vector cannot look the type up until the extension exists, so this
    has to happen before the pool opens. It deliberately does not run inside the
    pool's configure hook: CREATE EXTENSION IF NOT EXISTS is not race-safe, and
    several connections opening at once would collide on pg_extension_name_index.
    """
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _ensure_extension()
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=8,
            configure=_configure,
            open=True,
        )
    return _pool


@contextmanager
def connection():
    with pool().connection() as conn:
        yield conn


def init_schema() -> None:
    with connection() as conn:
        conn.execute(SCHEMA % {"dim": settings.embed_dim})
        conn.commit()


def reset() -> None:
    """Drop everything. Used by the ingestion script with --reset."""
    with connection() as conn:
        conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
        conn.execute("DROP TABLE IF EXISTS documents CASCADE")
        conn.commit()
    init_schema()
