# 0002 — MongoDB time-series window, idempotent by high-water mark

**Status:** Superseded (2026-10-10, trim) — MongoDB and the Parquet store were removed; readings live in a Postgres table whose primary key (location_id, ts) gives idempotency with `ON CONFLICT DO NOTHING`. See [0009](0009-trim-for-defensibility.md).

## Decision
Recent readings live in a MongoDB time-series collection (`timeField: ts`, `metaField: meta`, `granularity: hours`, `expireAfterSeconds` = window). Time-series collections cannot have unique indexes, so a poll is made idempotent differently: before inserting, the poller reads the newest stored timestamp for each (source, location, parameter) series and inserts only newer readings. Full history goes to Parquet files in S3 (append-only per poll run; readers dedupe).

## Alternatives
- Regular collection with a unique index on (source, location, parameter, ts): simpler idempotency, but loses time-series compression and automatic bucketing.
- Upserts per reading: one round trip per reading, and not supported the same way on time-series collections.

## Consequences
- ✅ Re-running a poll inserts nothing (tested). ≈ 94 bytes per reading incl. indexes, so the 512 MB Atlas M0 budget is not a constraint.
- ⚠️ A late reading older than the series' high-water mark is dropped from the window (it still reaches S3).
