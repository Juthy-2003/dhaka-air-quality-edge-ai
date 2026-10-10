# 0008 — One Dockerfile, `worker` and `api` targets

**Status:** Superseded (2026-10-10, trim) — AWS Lambda was removed; there is one app image (FastAPI, no Lambda adapter) and the separate edge image. See [0009](0009-trim-for-defensibility.md).

## Decision
The Lambda Web Adapter extension, once copied to `/opt/extensions`, starts in every Lambda function built from that image and waits for a web server. The poller is not a web server. So the Dockerfile builds `worker` (Lambda runtime interface client; also used for the simulator and RAG service) and `api` (worker + adapter + gunicorn); SAM selects them with `DockerBuildTarget`. The edge gateway has its own small image (`Dockerfile.edge`, 412 MB: onnxruntime, numpy, paho-mqtt, requests) with no scikit-learn, pandas or MLflow.

## Consequences
- ✅ One build context, no adapter where it would hang a function.
- ⚠️ The app image is 1.7 GB (fastembed, onnxruntime, scikit-learn, MLflow client); a slimmer RAG-only image is possible later.
