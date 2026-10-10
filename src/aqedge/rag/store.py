"""pgvector store: documents and chunks with 384-d embeddings, cosine
distance, HNSW index. Re-ingesting a corpus version replaces it.

Same Postgres database as the readings and anomalies (aqedge/db.py)."""

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from aqedge import config
from aqedge.rag.embed import DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    doc_id text PRIMARY KEY, url text, title text, publisher text,
    licence text, licence_checked boolean, fetched_at timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id text PRIMARY KEY, doc_id text REFERENCES documents, page int, text text,
    embedding vector({DIM}), corpus_version text
);
CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS guidance (
    category text PRIMARY KEY, advice text, doc_id text REFERENCES documents, page int, corpus_version text
);
"""


def connect():
    conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load(sources, chunks, embeddings: np.ndarray, corpus_version: str, guidance: dict | None = None) -> int:
    with connect() as conn:
        conn.execute(SCHEMA)
        conn.execute("TRUNCATE chunks, guidance")
        for s in sources:
            conn.execute(
                "INSERT INTO documents (doc_id, url, title, publisher, licence, licence_checked) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (doc_id) DO UPDATE SET url = EXCLUDED.url, "
                "title = EXCLUDED.title, licence = EXCLUDED.licence, fetched_at = now()",
                (s.doc_id, s.url, s.title, s.publisher, s.licence, s.licence_checked),
            )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (chunk_id, doc_id, page, text, embedding, corpus_version) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (c.chunk_id, c.doc_id, c.page, c.text, e, corpus_version)
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )
            cur.executemany(
                "INSERT INTO guidance (category, advice, doc_id, page, corpus_version) "
                "VALUES (%s, %s, 'airnow-pm-guide', 1, %s)",
                [(cat, advice, corpus_version) for cat, advice in (guidance or {}).items()],
            )
    return len(chunks)


def search(query_vec: np.ndarray, k: int = 5) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.chunk_id, c.doc_id, c.page, c.text, d.title, d.url, 1 - (c.embedding <=> %s) AS similarity "
            "FROM chunks c JOIN documents d USING (doc_id) ORDER BY c.embedding <=> %s LIMIT %s",
            (query_vec, query_vec, k),
        ).fetchall()
    keys = ["chunk_id", "doc_id", "page", "text", "title", "url", "similarity"]
    return [dict(zip(keys, r, strict=True)) for r in rows]


def corpus_version() -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT corpus_version FROM chunks LIMIT 1").fetchone()
    return row[0] if row else None


def category_guidance(category: str) -> dict | None:
    """The advice row for an AQI category, parsed from the AirNow guide at ingest."""
    with connect() as conn:
        row = conn.execute(
            "SELECT g.advice, g.doc_id, g.page, d.title, d.url FROM guidance g JOIN documents d USING (doc_id) "
            "WHERE g.category = %s",
            (category,),
        ).fetchone()
    if not row:
        return None
    advice, doc_id, page, title, url = row
    return {
        "chunk_id": f"{doc_id}:p{page}:advice:{category}",
        "doc_id": doc_id,
        "page": page,
        "text": f"Advice for {category}: {advice}",
        "title": title,
        "url": url,
        "similarity": 1.0,
    }
