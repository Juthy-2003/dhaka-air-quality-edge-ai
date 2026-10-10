# Retrieval eval set

`questions.jsonl`: 32 questions, each with **evidence phrases** — short exact
strings from the corpus that a relevant chunk must contain. A retrieved chunk
counts as relevant if it contains any of the question's phrases
(case-insensitive). Labelling by phrase instead of chunk id means the labels
survive changes to chunk size and overlap, which is exactly what the eval
gate is meant to compare.

**Status: first draft (2026-10-10), written from the fetched corpus. Every
label still needs a second review, and the set should grow to the planned
50–100 questions.**

Metrics (`python -m aqedge.rag.evaluate`): recall@5, MRR@5. The CI gate fails
if either drops more than 0.05 below `baseline.json`.
