# 0009 — Trim for defensibility: edge + RAG only

**Status:** Accepted (2026-10-10). Supersedes 0002, 0005 and 0008; updates 0001, 0003 and 0006.

## Context
The first version had three parts: serverless ingestion on AWS (Lambda poller, Parquet in S3, MongoDB time-series, Django Ninja API), an edge gateway (ONNX model, MQTT, OTA from Project 2's MLflow, plus an INT8 autoencoder study) and a RAG assistant on Kubernetes. That is too much for a fresh graduate to learn well enough to defend every line. The backend skills (Django, MongoDB, AWS serverless) are already proven by a sibling project. What this project must prove for the MLOps job is: edge deployment, a sensor time-series stream, LLM + RAG, Docker, and CI/CD for ML.

## Decision
Keep the edge gateway and the RAG assistant; replace the ingestion/backend half with one small FastAPI app on the same Postgres the RAG corpus already used.

| Removed | Why | Replaced by |
|---|---|---|
| Django + Django Ninja API, migrations, `manage.py`, gateway model with peppered key hashes | Backend skills proven elsewhere; a second web framework next to FastAPI | `src/aqedge/app.py` (FastAPI): `/locations`, `/locations/{id}/latest`, `/anomalies`, `/ask`, `/models/manifest.json`; one gateway key from the environment |
| MongoDB time-series window + high-water-mark idempotency (ADR 0002) | A second database to learn | Postgres table `readings`, primary key (location_id, ts), `ON CONFLICT DO NOTHING` |
| Parquet raw store (local / S3, moto tests) | No consumer of the history in this project | — (the replay dataset itself is the history) |
| AWS Lambda poller + local loop, SAM template, `deploy-aws.yml`, Lambda Web Adapter (ADR 0008) | Never deployed; AWS proven elsewhere | A background thread in the app replays data every 15 min |
| OpenAQ v3 client | Never used (no key); 67 lines for nothing | Replay is the only source (ADR 0001) |
| MLflow registry REST flow + `register_edge_model.py` (ADR 0005) | A dependency on Project 2's MLflow for two REST calls | A static `models/manifest.json` {version, url, sha256, threshold} written by `scripts/publish_model.py` and served by the app |
| Autoencoder, INT8 quantisation, FP32-vs-INT8 benchmark, hybrid drift experiment | Not shipped; the INT8 finding is a story, not running code | A history note in BENCHMARKS.md (measured before the trim, labelled as such) |
| Helm chart, k3d instructions, Traefik notes | Kubernetes is not needed to run one assistant | docker compose on one machine (or one VM) |
| `images.yml` (multi-arch + QEMU) | Build complexity; QEMU timings were never usable anyway | Local `docker compose build` |

Kept unchanged in behaviour: fault injection, features (incl. peer deviation), rules baseline, isolation forest exported to ONNX, MQTT QoS 1 with dedupe, SQLite store-and-forward outbox, OTA checksum → smoke test → atomic swap → rollback, the RAG relevance gate, place resolution, facts-in-code, category advice as source [1], computed citations and the extractive fallback.

Added: a throughput benchmark on the edge path (`bench/edge_throughput.py`), so "at scale" is measured, not claimed.

## Consequences
- ✅ 6 compose services instead of 9 (in 3 profiles) plus a Helm chart; Python excluding tests 3,276 → 2,553 lines (57 → 34 files); compose + workflow YAML 569 → 136 lines (Helm and SAM gone); runtime dependencies 23 → 12; app image 1.69 GB → 869 MB.
- ✅ Measured "at scale": the 0.5-CPU gateway keeps up with 250 messages/s with no loss and p95 latency 3.3 ms, and saturates at about 445 messages/s (BENCHMARKS.md §3).
- ✅ No dependency on Project 2's MLflow or k3d cluster.
- ⚠️ No model registry: version history is the `models/published/vN/` folders, and there is no UI or metric lineage. A registry is worth it with many models or many people.
- ⚠️ One shared gateway key instead of one hashed key per gateway; fine for one gateway, not for a fleet.
- ⚠️ No raw history store and no live source; adding OpenAQ later means writing a client that produces the same rows.
