# 0005 — Over-the-air updates from MLflow's REST API

**Status:** Superseded (2026-10-10, trim) — MLflow was removed from this project; the gateway now polls a static `models/manifest.json` {version, url, sha256, threshold} served by the app. The checksum → smoke test → rename → rollback steps are unchanged. See [0009](0009-trim-for-defensibility.md).

## Decision
The gateway polls `GET /api/2.0/mlflow/registered-models/alias?name=aq-anomaly&alias=champion`. On a new version it downloads `edge/meta.json` and the ONNX file it names (`/get-artifact`), checks the file's sha256 against meta.json, runs a smoke test on a fixed 240-row sample (finite scores; ≤ 20% alarms on normal rows; ≥ 50% on injected spikes), swaps directories with `rename()` keeping `.previous`, and reloads; if the reload fails it renames back. It uses `requests` only — no MLflow client on the gateway. Models are registered with `create_model_version(source=<run>/edge)`, because MLflow 3's `register_model("runs:/…")` expects an MLflow "logged model", not plain files.

## Findings
- Renaming a directory baked into the image fails on overlay filesystems ("Invalid cross-device link"); the agent now copies its bundled model to a writable volume on first boot.
- An exception escaping one update killed the updater thread; the loop now survives any error.

## Consequences
- ✅ Tested: swap, checksum rejection, broken model never replaces the running one, rollback.
- ✅ Demonstrated: local → v1 → v2 on a running gateway.
- ⚠️ Polling every 20–30 s; a push channel (MQTT retained message) would be lighter at fleet scale.
