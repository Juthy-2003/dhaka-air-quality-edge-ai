"""The RAG assistant behind POST /ask (served by aqedge/app.py).

1. Relevance gate: if no guidance chunk is similar enough, refuse without
   calling the LLM.
2. Resolve the place in the question to a monitored location (deterministic).
3. Look up its latest reading in the readings table — a fixed "tool" called
   in code, because small models are unreliable at choosing tools themselves.
4. Code writes the facts sentence (value, place, age, US EPA category); the
   advice row for that category is source [1], plus retrieved chunks.
5. The local LLM (llama.cpp, OpenAI-compatible API) writes 2–3 sentences of
   advice; citations are attributed by embedding similarity, unsupported
   sentences are flagged, and an extractive fallback quotes the advice row
   if the model still refuses.

It is a demo, not health advice, and says so.
"""

import json
import os
import re
import time

import numpy as np
import requests
from prometheus_client import Counter, Histogram

from aqedge import db
from aqedge.rag import store
from aqedge.rag.aqi import pm25_category
from aqedge.rag.embed import embed_passages, embed_query
from aqedge.rag.locations import resolve

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8081")
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
CHUNK_CHARS_IN_PROMPT = 600  # prompt prefill dominates latency on a CPU (~25–30 s for 5 × 900 chars)
SUPPORT_THRESHOLD = 0.6  # cosine similarity for a sentence to count as supported by a source
# Relevance gate: if even the best chunk is this dissimilar, the question is
# off-topic and the LLM is not called. Measured 2026-10-10 on 41 in-scope and
# 9 off-topic questions: lowest in-scope 0.499, highest off-topic 0.492 — a
# thin margin, so treat this as a tuned-on-small-data guardrail, not a proof.
RELEVANCE_THRESHOLD = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.495"))
GATED = Counter("rag_relevance_gate_refusals_total", "Questions refused before calling the LLM")
PROMPT_VERSION = "ask-v4"  # v4: facts written by code; model only summarises advice; extractive fallback
DISCLAIMER = "This is not medical advice."
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "qwen2.5-1.5b-instruct-q4_k_m")

# Two prompts. With a reading, relevant advice is always source [1], so the
# model gets no refusal option (a 1.5B model takes that exit far too often);
# without one, refusing is the right answer to off-topic questions.
SYSTEM_WITH_READING = (
    "You give air-quality advice to the public. The current category is given. Using ONLY source [1] "
    "(and the other sources if relevant), tell the person in 2-3 short sentences what they should do for "
    "their question, including advice for sensitive groups such as children, older adults and people with asthma "
    "when relevant. Do not repeat the reading. Cite sources like [1]."
)
SYSTEM_GENERAL = (
    "You answer questions about air quality using ONLY the numbered sources. Cite sources like [1]. "
    "If none of the sources relate to the question, reply exactly: The sources I have do not cover this. "
    "Answer in at most 4 short sentences."
)
REFUSAL = "The sources I have do not cover this"

ASKS = Counter("rag_requests_total", "Questions answered", ["outcome"])
TTFT = Histogram(
    "rag_llm_time_to_first_token_seconds", "LLM time to first token", buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16)
)
TPS = Histogram("rag_llm_tokens_per_second", "LLM generation speed", buckets=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64))
RETRIEVAL = Histogram("rag_retrieval_seconds", "Embedding + vector search", buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5))
TOTAL = Histogram("rag_request_seconds", "End-to-end /ask latency", buckets=(0.5, 1, 2, 4, 8, 16, 32, 64))


def latest_reading(location: str) -> dict | None:
    """Latest PM2.5 for a monitored location, with its US EPA category."""
    row = db.latest(location=location)
    if row is None or row["pm25"] is None:
        return None
    return {
        "parameter": "pm25",
        "value": round(float(row["pm25"]), 1),
        "unit": row["unit"],
        "ts": row["ts"],
        "age_minutes": row["age_minutes"],
        "stale": row["stale"],
        "source": row["source"],
        "replayed_from": row["replayed_from"],
        "location": location,
        "category": pm25_category(row["pm25"]),
    }


def generate(messages: list[dict]) -> tuple[str, dict]:
    """Stream from llama-server to measure time-to-first-token and tokens/s."""
    t0 = time.perf_counter()
    first, text, n_tokens = None, [], 0
    with requests.post(
        f"{LLM_URL}/v1/chat/completions",
        stream=True,
        timeout=180,
        json={"messages": messages, "temperature": 0.1, "max_tokens": 220, "stream": True},
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta:
                first = first or time.perf_counter()
                text.append(delta)
                n_tokens += 1
    end = time.perf_counter()
    ttft = (first or end) - t0
    tps = n_tokens / (end - first) if first and end > first else 0.0
    TTFT.observe(ttft)
    TPS.observe(tps)
    return "".join(text).strip(), {"ttft_s": round(ttft, 3), "tokens": n_tokens, "tokens_per_s": round(tps, 1)}


def build_messages(question: str, reading: dict | None, chunks: list[dict]) -> list[dict]:
    parts = []
    if reading:
        parts.append(f"Current category: {reading['category']} (PM2.5 {reading['value']} {reading['unit']}).")
    parts.append("Sources:")
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] ({c['title']}, p.{c['page']}) {c['text'][:CHUNK_CHARS_IN_PROMPT]}")
    system = SYSTEM_WITH_READING if reading else SYSTEM_GENERAL
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts) + f"\n\nQuestion: {question}"},
    ]


def attribute(answer: str, chunks: list[dict]) -> tuple[list[int], list[dict]]:
    """Attribute each answer sentence to its most similar source (cosine on bge
    embeddings). A 1.5B model rarely writes [n] reliably, so citations are
    computed rather than trusted; sentences below SUPPORT_THRESHOLD are
    reported as unsupported."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\[\d+\]", "", answer)) if len(s.strip()) > 20]
    sentences = [s for s in sentences if DISCLAIMER not in s]
    if not sentences or not chunks:
        return [], []
    sv = embed_passages(sentences)
    cv = embed_passages([c["text"][:CHUNK_CHARS_IN_PROMPT] for c in chunks])
    sims = sv @ cv.T / (np.linalg.norm(sv, axis=1)[:, None] * np.linalg.norm(cv, axis=1)[None, :])
    support, cited = [], set()
    for i, s in enumerate(sentences):
        j = int(sims[i].argmax())
        ok = float(sims[i, j]) >= SUPPORT_THRESHOLD
        support.append({"sentence": s, "source": j + 1 if ok else None, "similarity": round(float(sims[i, j]), 3)})
        if ok:
            cited.add(j + 1)
    return sorted(cited), support


def _response(
    answer,
    *,
    chunks,
    place,
    reading,
    llm,
    t0,
    citations=(),
    model_cited=(),
    support=(),
    refused=False,
    fallback=False,
    gated=False,
) -> dict:
    return {
        "answer": answer,
        "citations": [
            {
                "n": n,
                "chunk_id": chunks[n - 1]["chunk_id"],
                "title": chunks[n - 1]["title"],
                "page": chunks[n - 1]["page"],
                "url": chunks[n - 1]["url"],
            }
            for n in citations
        ],
        "invalid_citations": [n for n in model_cited if not 1 <= n <= len(chunks)],
        "citations_from_model": list(model_cited),
        "sentence_support": list(support),
        "refused": refused,
        "extractive_fallback": fallback,
        "gated": gated,
        "place": place[0] if place else None,
        "reading_used": reading,
        "retrieved": [{"chunk_id": c["chunk_id"], "similarity": round(float(c["similarity"]), 3)} for c in chunks],
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "corpus_version": store.corpus_version(),
        "llm": llm,
        "seconds": round(time.perf_counter() - t0, 2),
        "disclaimer": "Demo assistant, not medical advice.",
    }


def ask(question: str) -> dict:
    t0 = time.perf_counter()
    place = resolve(question)

    # 1. Relevance gate: off-topic questions are refused without calling the LLM.
    t = time.perf_counter()
    chunks = store.search(embed_query(question), TOP_K)
    RETRIEVAL.observe(time.perf_counter() - t)
    if not chunks or chunks[0]["similarity"] < RELEVANCE_THRESHOLD:
        GATED.inc()
        ASKS.labels("refused").inc()
        no_llm = {"ttft_s": 0.0, "tokens": 0, "tokens_per_s": 0.0}
        return _response(
            f"{REFUSAL}. {DISCLAIMER}",
            chunks=chunks,
            place=place,
            reading=None,
            llm=no_llm,
            t0=t0,
            refused=True,
            gated=True,
        )

    # 2. The reading (a fixed tool call) and its category's advice as source [1].
    reading = latest_reading(place[1]) if place else None
    if reading:
        query = f"{question} air quality {reading['category']}"
        chunks = store.search(embed_query(query), TOP_K)
        guide = store.category_guidance(reading["category"])
        if guide:
            chunks = [guide] + chunks[: TOP_K - 1]

    # 3. Generate; facts come from code, the model only summarises advice.
    text, llm = generate(build_messages(question, reading, chunks))
    fallback = False
    if reading:
        lead = (
            f"Near {place[0]}, PM2.5 is {reading['value']} {reading['unit']} (measured at {reading['location']}, "
            f"{reading['age_minutes']:.0f} minutes ago{'; this reading is STALE' if reading['stale'] else ''}), "
            f'which is "{reading["category"]}" on the US EPA scale (approximate: one hourly value).'
        )
        if (not text or text.startswith(REFUSAL)) and chunks[0]["chunk_id"].endswith(reading["category"]):
            text, fallback = f"{chunks[0]['text']} [1]", True  # extractive fallback: quote the advice row
        answer = f"{lead} {text}"
    else:
        answer = text

    # 4. Citations: what the model wrote, plus attribution by similarity.
    refused = answer.startswith(REFUSAL)
    model_cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    attributed, support = ([], []) if refused else attribute(answer, chunks)
    citations = sorted({n for n in model_cited if 1 <= n <= len(chunks)} | set(attributed))
    if DISCLAIMER not in answer:
        answer = f"{answer} {DISCLAIMER}"
    ASKS.labels("refused" if refused else ("ok" if citations else "unsupported")).inc()
    TOTAL.observe(time.perf_counter() - t0)
    return _response(
        answer,
        chunks=chunks,
        place=place,
        reading=reading,
        llm=llm,
        t0=t0,
        citations=citations,
        model_cited=model_cited,
        support=support,
        refused=refused,
        fallback=fallback,
    )
