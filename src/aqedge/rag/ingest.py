"""Fetch the corpus, chunk it, embed it, load pgvector.

uv run python -m aqedge.rag.ingest      # DATABASE_URL defaults to the compose Postgres on :5435
"""

import json

from aqedge.rag import chunk as C
from aqedge.rag import store
from aqedge.rag.corpus import SOURCES, corpus_hash, fetch
from aqedge.rag.embed import embed_passages
from aqedge.rag.guidance import parse as parse_guidance


def main() -> dict:
    pages = [p for s in SOURCES for p in fetch(s)]
    chunks = C.chunk_pages(pages)
    version = corpus_hash(
        SOURCES,
        {"size": C.CHUNK_CHARS, "overlap": C.OVERLAP_CHARS, "chunker": C.CHUNKER_VERSION},
        [p.text for p in pages],
    )
    guide_page = next(p for p in pages if p.doc_id == "airnow-pm-guide" and p.page == 1)
    guidance = parse_guidance(guide_page.text)
    n = store.load(SOURCES, chunks, embed_passages([c.text for c in chunks]), version, guidance)
    return {
        "documents": len(SOURCES),
        "pages": len(pages),
        "chunks": n,
        "guidance_categories": len(guidance),
        "corpus_version": version,
    }


if __name__ == "__main__":
    print(json.dumps(main()))
