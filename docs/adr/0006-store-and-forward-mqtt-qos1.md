# 0006 — MQTT QoS 1, SQLite store-and-forward, idempotent API

**Status:** Accepted (2026-10-10)

> **Update (2026-10-10, trim, [0009](0009-trim-for-defensibility.md)):** "the API" is now the FastAPI app; idempotency comes from the `anomalies` table's primary key `event_id` (`ON CONFLICT DO NOTHING`), and the gateway authenticates with one shared key from the environment instead of per-gateway peppered hashes.

## Decision
Sensors publish to `aq/<site>/<sensor>/<parameter>` with QoS 1 (at least once); the agent drops redeliveries by (sensor, seq). Anomalies go into a SQLite outbox first and are deleted only after the API acknowledges them; a failed send backs off (1 → 30 s). Event ids are deterministic (uuid5 of gateway/sensor/seq), and the API is idempotent on (gateway, event_id), so a resend after a lost acknowledgement creates no duplicate.

## Alternatives
QoS 2 (exactly once) costs an extra handshake per message and is unnecessary when the consumer deduplicates; sending directly without a queue loses events during outages.

## Consequences
- ✅ Measured: a ~20 s outage queued 114 events (max 33 s) and all arrived; nothing lost. Re-measured after the trim (2026-10-10, 100 messages/s): 32 events waited (max 31.6 s), 209 of 209 arrived, none duplicated.
- ⚠️ Found by the throughput benchmark: QoS 1 protects the hop, not a slow consumer. If the agent falls behind, Mosquitto queues at most 1,000 messages for it (`max_queued_messages` default) and silently drops the rest — at 500 messages/s on 0.5 CPU, 133 of 10,000 were dropped.
- ⚠️ The outbox is unbounded; a very long outage on a small disk needs a cap and a policy (drop oldest normals first).
