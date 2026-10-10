"""Answer-level check of the running assistant (not part of CI: needs the LLM).

In-scope questions must be answered (not refused) with at least one citation;
off-topic questions must be refused. Reports the share of answer sentences
supported by a source (cosine ≥ 0.6), time to first token and tokens/s.

    uv run python bench/rag_answers.py --url http://localhost:8000
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import requests

IN_SCOPE = [
    "Is it safe to go jogging in Mirpur this evening?",
    "Can my kids play outside in Uttara today?",
    "My grandmother lives in Gulshan, should she go for a walk?",
    "I have asthma and live in Dhanmondi. Anything I should do today?",
    "What is the NowCast?",
    "Who counts as a sensitive group for particle pollution?",
    "How can I reduce particle pollution inside my home?",
    "Where does PM2.5 come from?",
]
OFF_TOPIC = [
    "Who won the cricket match yesterday?",
    "What is the capital of France?",
    "Write me a poem about the sea.",
    "How do I bake a chocolate cake?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    rows = []
    for q, expect in [(q, "answer") for q in IN_SCOPE] + [(q, "refuse") for q in OFF_TOPIC]:
        t = time.perf_counter()
        d = requests.post(f"{a.url}/ask", json={"question": q}, timeout=300).json()
        sup = d["sentence_support"]
        rows.append(
            {
                "question": q,
                "expect": expect,
                "refused": d["refused"],
                "fallback": d["extractive_fallback"],
                "citations": len(d["citations"]),
                "supported_sentences": sum(s["source"] is not None for s in sup),
                "sentences": len(sup),
                "gated": d.get("gated", False),
                "ttft_s": d["llm"]["ttft_s"],
                "tokens_per_s": d["llm"]["tokens_per_s"],
                "seconds": round(time.perf_counter() - t, 2),
                "answer": d["answer"],
            }
        )
    correct = [
        r for r in rows if (r["refused"] if r["expect"] == "refuse" else (not r["refused"] and r["citations"] > 0))
    ]
    sents = sum(r["sentences"] for r in rows)
    llm_rows = [r for r in rows if not r["gated"]]  # gate refusals never call the LLM
    ttft = sorted(r["ttft_s"] for r in llm_rows)
    summary = {
        "questions": len(rows),
        "behaved_as_expected": len(correct),
        "in_scope_answered_with_citation": sum(
            1 for r in rows if r["expect"] == "answer" and not r["refused"] and r["citations"]
        ),
        "off_topic_refused": sum(1 for r in rows if r["expect"] == "refuse" and r["refused"]),
        "extractive_fallbacks": sum(r["fallback"] for r in rows),
        "supported_sentence_share": round(sum(r["supported_sentences"] for r in rows) / sents, 3) if sents else None,
        "ttft_p50_s": statistics.median(ttft),
        "ttft_max_s": ttft[-1],
        "tokens_per_s_median": statistics.median(r["tokens_per_s"] for r in llm_rows if r["tokens_per_s"]),
        "end_to_end_median_s_llm_calls": statistics.median(r["seconds"] for r in llm_rows),
        "refused_by_relevance_gate": sum(r["gated"] for r in rows),
        "llm_calls": len(llm_rows),
        "note": "llama-server restarted before the run (cold prompt cache); the shared system prompt is reused "
        "from the server's prompt cache after the first call",
        "setup": "Qwen2.5-1.5B-Instruct Q4_K_M, llama.cpp server (2 threads, 2 CPUs, 2 GB, docker on Apple M4), concurrency 1",
        "machine": platform.platform(),
        "measured_at": time.strftime("%Y-%m-%d"),
        "readings": "replay-mendeley (historical, modelled values replayed as live)",
    }
    Path("bench/results/rag_answers.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
