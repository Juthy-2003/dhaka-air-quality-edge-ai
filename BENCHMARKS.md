# Benchmarks

Every number here was measured on **2026-10-10**, after the trim to edge + RAG (ADR 0009). Method and raw output are in `bench/`, `scripts/` and `bench/results/`. Host: an Apple M4 laptop (macOS, arm64) running Docker; **no number comes from a Raspberry Pi, an ARM server or an x86 machine**, because none was available. Sensor data is SYNTHETIC: modelled historical values (see README) with injected faults.

## 1. Anomaly detection (offline, test year 2025)

`scripts/train_anomaly.py` → `bench/results/anomaly.json`. 8 sensor series (PM2.5 and PM10 at 4 Dhaka locations); faults injected with a fixed seed: 224 faults in the test year (stuck, spike, drift, dropout). Train 2022-08 → 2023 (clean), threshold picked on 2024 (separate seed), reported once on 2025. Event-level scoring, ±2 readings tolerance.

| Detector | Precision | Recall | F1 | Stuck | Spike | Drift | Dropout | False-alarm episodes |
|---|---|---|---|---|---|---|---|---|
| Rules (baseline) | 0.33 | 0.90 | 0.48 | 1.00 | 0.98 | 0.61 | 1.00 | 478 |
| **Isolation forest (shipped, ONNX 1.74 MB)** | **0.90** | **0.75** | **0.82** | 0.93 | 0.95 | 0.09 | 1.00 | 20 |

Re-run after the trim: identical to the pre-trim run (same seeds, same model file, sha256 `112e68cb…`).

- **Slow drift is not solved**: 9% recall. Peer-deviation features (a sensor diverging from its neighbours) moved the drift rule from 0.95 recall with 7,543 false readings to 0.61 with far fewer. Next step: longer windows or a seasonal baseline per sensor.
- Dropout is always a rule (a null reading cannot be scored).

## 2. Edge gateway, end to end

`scripts/edge_demo.sh` → `scripts/edge_report.py` → `bench/results/edge_demo.json`. 16,000 readings from 8 SYNTHETIC sensors (2,000 simulated hours) at 100 messages/s over MQTT into the agent (0.5 CPU / 256 MB). The app (the uplink) is stopped for 20 s, 40 s into the stream; model v2 is published to the manifest 3 s after the uplink returns.

| Measure | Result |
|---|---|
| Readings received | 16,000 of 16,000 |
| Events forwarded | 209 (= 209 detected by the agent; none lost, none duplicated) |
| **Bandwidth saved** | **97.2%** (3.01 MB if every reading were forwarded, 84 KB sent) |
| Precision / recall (from what the gateway sent) | 1.00 / 0.72 — stuck 13/16, spike 12/14, drift 0/8, dropout 8/8 |
| Outage | 32 events waited in the SQLite outbox (max 31.6 s); all arrived |
| OTA | local → v1 before the stream; v1 → v2 **during** the stream: 85 events scored by v1, 68 by v2, 56 by the dropout rule. v2 is the same weights re-published — the demo exercises delivery, checksum, smoke test and swap, not a better model |
| Agent latency (publish → scored) | p50 2.8 ms, p95 38 ms |

The pre-trim demo (400 messages/s requested, MLflow OTA) measured 96.8% bandwidth saved, P 1.00 / R 0.72 and 114 events in the outbox. Bandwidth moved to 97.2% because each message now carries a `published_at` timestamp (bigger messages, same forwarded events); the outbox count differs because the outage now starts later in the stream.

## 3. Edge throughput ("at scale")

`bench/edge_throughput.py` → `bench/results/edge_throughput.json`. SYNTHETIC data. For each offered rate, the agent is restarted with empty state, the simulator (uncapped) publishes 20 s worth of messages (QoS 1), and the agent's own counters give received messages, achieved rate, latency from simulator publish to "scored" (p50/p95) and mean model time per reading; `docker stats` is sampled every second for peak memory and CPU. Same clock for both ends (one Docker VM).

| Offered msg/s | Received / published | Achieved msg/s | Latency p50 | Latency p95 | Model ms/reading | Agent peak CPU | Agent peak memory |
|---|---|---|---|---|---|---|---|
| **250** | **5,000 / 5,000** | **250** | **1.7 ms** | **3.3 ms** | 1.38 | 44% | 64 MiB |
| 500 | 9,867 / 10,000 | 437 | 1,183 ms | 2,528 ms | 1.97 | 51% | 66 MiB |
| 1,000 | 9,932 / 20,000 | 443 | 2,313 ms | 2,641 ms | 1.91 | 48% | 65 MiB |
| 2,000 | 9,972 / 40,000 | 446 | 2,301 ms | 2,494 ms | 1.90 | 50% | 64 MiB |
| 4,000 | 7,390 / 56,000 | 453 | 2,290 ms | 2,389 ms | 1.97 | 51% | 64 MiB |

(CPU % is of one core; the container's cap is 50%.)

- **Sustained: 250 messages/s** with no loss and p95 3.3 ms. **Capacity: about 445 messages/s**, CPU-bound at the 0.5-CPU cap. For scale: the demo's 8 sensors reporting hourly send 8 messages per *hour*; 250 messages/s is, for example, 1,000 sensors each reporting every 4 s (arithmetic, not a fleet test).
- **What saturation looks like:** above capacity, latency plateaus at ~2.3 s and messages are dropped. That is Mosquitto's per-client queue (`max_queued_messages`, default 1,000 messages ≈ 2.3 s at 445/s): once it is full the broker silently drops new QoS 1 messages for the slow subscriber. QoS 1 protects the hop, not a consumer that cannot keep up.
- **Headroom matters:** at 250 messages/s the agent already uses ~44% of its 50% budget. A first demo run at 250 messages/s lost 903 of 16,000 readings, because outbox draining after the outage plus the OTA smoke test pushed the agent over its budget. The demo now runs at 100 messages/s (~22% of capacity) and loses nothing.
- Model time per reading in the agent (1.4–2.0 ms) is higher than the pre-trim isolated microbenchmark (0.8–0.9 ms, one model in its own process) because it includes CPU-quota throttling and runs next to MQTT and feature code. The isolated number was not re-measured: that benchmark was removed with the INT8 code.

## 4. Retrieval (CI gate)

`python -m aqedge.rag.evaluate`. 32 labelled questions (a first draft written from the corpus, still to be reviewed), evidence-phrase relevance, k = 5, bge-small-en-v1.5, pgvector HNSW cosine. Re-run after moving the corpus into the shared Postgres: same corpus version (`d02ece303c13`, 89 chunks), same result.

| Corpus / chunking | recall@5 | MRR |
|---|---|---|
| First chunker (bug: duplicated, mid-word chunks) — before the trim | 0.84 | 0.77 |
| **Fixed chunker, 900/150 (baseline)** | **0.91** | **0.80** |
| 250-character chunks — **gate FAILED** (`bench/results/rag_gate_regression.txt`, before the trim) | 0.81 | 0.70 |

Still missed: the PM2.5 breakpoint values (table text from the PDF; q004, q005) and one children's-risk question (q011).

## 5. Assistant answers and LLM serving

`bench/rag_answers.py` → `bench/results/rag_answers.json`. `/ask` on the FastAPI app; Qwen2.5-1.5B-Instruct Q4_K_M on llama.cpp `llama-server` (2 threads, 2 CPUs, 2 GB), concurrency 1, server restarted first (cold prompt cache). Readings are the replayed (modelled) data.

| Measure | After trim (2026-10-10) | Before trim |
|---|---|---|
| 8 in-scope questions answered with ≥ 1 citation | 8 / 8 | 8 / 8 |
| 4 off-topic questions refused | 4 / 4 (all by the relevance gate, without calling the LLM) | 4 / 4 |
| Answer sentences supported by a source (cosine ≥ 0.6) | 90.5% | 90.9% |
| Time to first token, LLM calls | p50 8.1 s, max 10.9 s | p50 8.8 s, max 11.0 s |
| Generation speed | 18.3 tokens/s (median) | 18.5 |
| End to end, LLM calls | 10.9 s (median) | 12.8 s |

Before the guardrails: the model refused every question that needed a reading, cited nothing, and wrote a poem when asked for one (CHANGELOG). The relevance gate's margin is thin (lowest in-scope similarity 0.499, highest off-topic 0.492 on 50 questions — measured before the trim; the embedding model and corpus are unchanged). The model still sometimes over-states caution ("It is not safe to go jogging") beyond its source.

## 6. Images

| Image | Size |
|---|---|
| `aq-app:local` (FastAPI app + simulator; no scikit-learn) | 869 MB (was 1.69 GB with Django, MLflow, scikit-learn) |
| `aq-edge:local` (gateway: onnxruntime, numpy, paho-mqtt, requests) | 408 MB |
| LLM model file (GGUF Q4_K_M) | 1.1 GB, not in an image |

## History: results that were removed with their code (measured before the trim, not re-measured)

- **INT8 made the tiny autoencoder bigger and slower.** An MLP autoencoder (~300 weights, F1 0.76) was quantised with ONNX Runtime dynamic INT8: the file grew 2.6 → 4.8 KB and single-reading latency rose 8.7 → 11.6 µs on the Mac (5.5 → 8.3 µs in the 0.5-CPU container), while F1 moved 0.762 → 0.759. Quantise/dequantise overhead beats the savings on such a small model; INT8 pays off for large MatMul/Conv models. The shipped isolation forest is a tree ensemble with no MatMul, so INT8 never applied to it. The autoencoder, the INT8 code and `bench/onnx_bench.py` were removed (ADR 0009).
- A "model OR peer-drift rule" hybrid tuned on 2024 did not generalise to 2025 (iforest+drift F1 0.80 vs 0.82); removed.
- The RAG stack on k3d next to Project 2 used 3.8 GiB on the node (LLM 1.47 GiB) and served 20.1 s to first token; Kubernetes was removed.
- MongoDB time-series storage cost ≈ 94 bytes per reading; MongoDB was removed.
