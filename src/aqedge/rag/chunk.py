"""Split pages into overlapping chunks without breaking words or repeating text.

1. Break each page into units: paragraphs; a paragraph longer than the chunk
   size is broken into lines, and a line longer than that into sentences.
2. Pack units greedily into chunks of at most CHUNK_CHARS.
3. Start each new chunk with the last few units of the previous one (up to
   OVERLAP_CHARS) so a sentence near a boundary appears in both.

Chunk size and overlap are configuration, tuned against retrieval recall@5
(rag/evals); the defaults are what the recorded baseline used.
(The first version hard-split long paragraphs and re-appended its overlap,
producing duplicated, mid-word chunks — found 2026-10-10 when a PDF page
with no blank lines became one 2,600-character "paragraph".)
"""

import os
import re
from dataclasses import dataclass

from aqedge.rag.corpus import Page

CHUNKER_VERSION = "2"  # part of the corpus version: changing the code changes the corpus
CHUNK_CHARS = int(os.environ.get("RAG_CHUNK_CHARS", "900"))
OVERLAP_CHARS = int(os.environ.get("RAG_OVERLAP_CHARS", "150"))
MIN_CHARS = 80


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str


def _units(text: str, size: int) -> list[str]:
    out: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) <= size:
            out.append(para)
            continue
        for line in (x.strip() for x in para.split("\n")):
            if not line:
                continue
            if len(line) <= size:
                out.append(line)
            else:
                out += [s for s in re.split(r"(?<=[.!?])\s+", line) if s]
    # A sentence longer than the chunk size is kept whole rather than cut mid-word.
    return out


def chunk_pages(pages: list[Page], size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    out: list[Chunk] = []
    for p in pages:
        current: list[str] = []
        n = 0

        for unit in _units(p.text, size):
            if current and len("\n".join(current)) + 1 + len(unit) > size:
                n = _emit(out, p, current, n)
                tail: list[str] = []  # overlap: whole trailing units, up to `overlap` chars
                for u in reversed(current):
                    if len("\n".join([u, *tail])) > overlap:
                        break
                    tail.insert(0, u)
                current = tail
            current.append(unit)
        if current:
            _emit(out, p, current, n)
    return out


def _emit(out: list[Chunk], page: Page, units: list[str], n: int) -> int:
    text = "\n".join(units).strip()
    if len(text) < MIN_CHARS:
        return n
    out.append(Chunk(f"{page.doc_id}:p{page.page}:c{n}", page.doc_id, page.page, text))
    return n + 1
