# Architecture Decision Records

| # | Decision | Status |
|---|---|---|
| [0001](0001-replay-as-live.md) | Replay historical (modelled) data as live | Accepted (updated by 0009) |
| [0002](0002-timeseries-highwater-idempotency.md) | MongoDB time-series window with a per-series high-water mark for idempotency | Superseded by 0009 |
| [0003](0003-isolation-forest-on-edge.md) | Ship the isolation forest; INT8 only where it applies | Accepted (INT8 code removed by 0009) |
| [0004](0004-peer-deviation-features.md) | Compare each sensor with its neighbours to separate drift from pollution | Accepted |
| [0005](0005-ota-from-mlflow-rest.md) | OTA from MLflow's REST API with checksum, smoke test and rollback | Superseded by 0009 (manifest) |
| [0006](0006-store-and-forward-mqtt-qos1.md) | MQTT QoS 1 + SQLite store-and-forward + idempotent API | Accepted (updated by 0009) |
| [0007](0007-grounding-a-small-llm.md) | Grounding a 1.5B model: facts in code, structured advice, computed citations, relevance gate | Accepted |
| [0008](0008-two-image-targets.md) | One Dockerfile, `worker` and `api` targets (Lambda Web Adapter only where needed) | Superseded by 0009 |
| [0009](0009-trim-for-defensibility.md) | Trim to edge + RAG: one FastAPI app, one Postgres, OTA from a manifest | Accepted |
