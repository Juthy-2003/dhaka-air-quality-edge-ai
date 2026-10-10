# 0007 — Grounding a 1.5B model

**Status:** Accepted (2026-10-10)

## Context
Qwen2.5-1.5B-Instruct (Q4_K_M) runs on 2 CPU threads in 2 GB, which fits the shared node. Measured problems with the first, prompt-only design: it refused every question that needed a reading ("The sources I have do not cover this"), almost never wrote citations, and wrote a poem when asked for one.

## Decision (ask-v4)
1. **Relevance gate** before the LLM: if the best chunk's cosine similarity is < 0.495, refuse without calling the model (in-scope minimum 0.499 vs off-topic maximum 0.492 on 50 questions — thin, recorded as such).
2. **Facts in code**: the first sentence (value, place, measuring location, age, category) is written by the service; the model never states numbers.
3. **Structured advice**: the AirNow particle-pollution table is parsed at ingest into per-category advice; the row for the reading's category is always source [1]. Vector search over PDF table text kept retrieving the wrong rows.
4. **Two prompts**: with a reading there is no refusal option (relevant advice is always present); without one, refusing is allowed.
5. **Computed citations**: each answer sentence is attributed to its most similar source (bge cosine ≥ 0.6); unsupported sentences are reported, not hidden.
6. **Extractive fallback**: if the model still declines, the service quotes the advice row.
7. The disclaimer is appended in code.

## Consequences
- ✅ 8/8 in-scope answered with citations, 4/4 off-topic refused, 91% of sentences supported (bench/results/rag_answers.json).
- ⚠️ The model sometimes over-states caution ("It is not safe to go jogging") beyond the source; the support check flags such sentences.
- ⚠️ ~9 s to first token on 2 threads, dominated by prompt prefill.
