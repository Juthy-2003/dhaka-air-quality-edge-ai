"""Retrieval evaluation and CI gate.

uv run python -m aqedge.rag.evaluate                 # print metrics
uv run python -m aqedge.rag.evaluate --gate          # exit 1 if below baseline − tolerance
uv run python -m aqedge.rag.evaluate --write-baseline
"""

import argparse
import json
import sys
from pathlib import Path

from aqedge.rag import store
from aqedge.rag.embed import embed_query

HERE = Path(__file__).parent / "evals"
TOLERANCE = 0.05


def relevant(chunk_text: str, evidence: list[str]) -> bool:
    t = chunk_text.lower().replace("’", "'")
    return any(e.lower().replace("’", "'") in t for e in evidence)


def evaluate(k: int = 5) -> dict:
    qs = [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines() if line.strip()]
    hits, rr, misses = 0, 0.0, []
    for q in qs:
        results = store.search(embed_query(q["question"]), k)
        ranks = [i for i, c in enumerate(results, 1) if relevant(c["text"], q["evidence"])]
        if ranks:
            hits += 1
            rr += 1 / ranks[0]
        else:
            misses.append(q["id"])
    return {
        "questions": len(qs),
        "k": k,
        "recall_at_k": round(hits / len(qs), 4),
        "mrr": round(rr / len(qs), 4),
        "misses": misses,
        "corpus_version": store.corpus_version(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    a = ap.parse_args()
    m = evaluate()
    print(json.dumps(m, indent=2))
    base_path = HERE / "baseline.json"
    if a.write_baseline:
        base_path.write_text(json.dumps(m, indent=2))
    elif a.gate:
        base = json.loads(base_path.read_text())
        bad = [k for k in ("recall_at_k", "mrr") if m[k] < base[k] - TOLERANCE]
        if bad:
            print(f"FAIL: {bad} dropped more than {TOLERANCE} below baseline {base}")
            sys.exit(1)
        print("PASS: retrieval within tolerance of baseline")


if __name__ == "__main__":
    main()
