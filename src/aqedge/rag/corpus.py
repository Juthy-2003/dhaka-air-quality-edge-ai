"""The guidance corpus: fetched at ingest time, never committed.

Licences are recorded per source and marked unverified until someone checks
them; chunks are not published in the repo (licences unverified).
"""

import hashlib
import io
import re
from dataclasses import dataclass
from html.parser import HTMLParser

import requests

UA = {"User-Agent": "Mozilla/5.0 (aqedge corpus fetcher; portfolio project)"}


@dataclass(frozen=True)
class Source:
    doc_id: str
    url: str
    title: str
    publisher: str
    licence: str
    licence_checked: bool = False


SOURCES = [
    Source(
        "epa-aqi-tad",
        "https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf",
        "Technical Assistance Document for the Reporting of Daily Air Quality – the Air Quality Index (AQI)",
        "US EPA",
        "US federal government work (public domain in the US) — unverified",
    ),
    Source(
        "airnow-pm-guide",
        "https://document.airnow.gov/air-quality-guide-for-particle-pollution.pdf",
        "Air Quality Guide for Particle Pollution",
        "US EPA / AirNow",
        "US federal government work (public domain in the US) — unverified",
    ),
    Source(
        "who-aq-factsheet",
        "https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health",
        "Ambient (outdoor) air pollution — fact sheet",
        "World Health Organization",
        "© WHO; reuse terms unverified — do not redistribute text",
    ),
]
# Not included: WHO global air quality guidelines 2021 (PDF blocks scripted
# downloads); Bangladesh Department of Environment material (no stable
# document URL found on 2026-10-10).


@dataclass
class Page:
    doc_id: str
    page: int
    text: str


class _Text(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in ("p", "li", "h1", "h2", "h3", "h4", "tr", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def fetch(src: Source, session=None) -> list[Page]:
    http = session or requests.Session()
    r = http.get(src.url, headers=UA, timeout=120)
    r.raise_for_status()
    if src.url.endswith(".pdf") or r.headers.get("content-type", "").startswith("application/pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(r.content))
        return [Page(src.doc_id, i + 1, _clean(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    parser = _Text()
    parser.feed(r.text)
    text = _clean("".join(parser.parts))
    start = text.find("Key facts")  # drop site navigation before the article
    end = text.find("\nRelated\n")  # and the related-links block after it
    return [Page(src.doc_id, 1, text[start if start > 0 else 0 : end if end > 0 else None])]


def corpus_hash(sources, chunk_cfg: dict, page_texts: list[str]) -> str:
    h = hashlib.sha256()
    for s in sources:
        h.update(s.url.encode())
    h.update(repr(sorted(chunk_cfg.items())).encode())
    for t in page_texts:
        h.update(t.encode())
    return h.hexdigest()[:12]
