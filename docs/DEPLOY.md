# Deploying to one VM

Nothing is deployed yet. The whole system runs with docker compose on any single Linux VM. An arm64 always-free VM is enough: the LLM needs 2 CPUs and 2 GB, and the rest needs well under 1 GB.

1. **VM.** Use Ubuntu with Docker Engine and the compose plugin, and at least 4 GB RAM and 10 GB disk. The images take about 1.3 GB, the LLM 1.1 GB and Postgres a few MB. Open only port 8000 to the internet (or put a reverse proxy with TLS in front). Never expose 5435 (Postgres), 8081 (llama.cpp) or 1883 (MQTT).
2. **Code and data.** Run `git clone`, then `make setup data train llm-model` on the VM. Alternatively, copy `data/raw/dhaka_aqi_hourly.parquet` and `data/llm/*.gguf` across.
3. **Secrets.** Edit `docker-compose.yml` or add an `.env` file:
   - Set `GATEWAY_KEY` to a long random value in both `app` and `agent`.
   - Change the Postgres password (`POSTGRES_PASSWORD` and the `DATABASE_URL` in `app`).
4. **Start.** Run `make up`, then `make ingest` (it fetches the guidance corpus). Check `curl localhost:8000/health`.
5. **Publish the first model.** Run `make publish`. The gateway picks it up within `OTA_POLL_SECONDS`.
6. **A real gateway** (Raspberry Pi or any arm64 box): build `Dockerfile.edge` and run it with these settings:
   - `MQTT_HOST`
   - `API_URL=https://<vm>`
   - `MODEL_MANIFEST_URL=https://<vm>/models/manifest.json`
   - `GATEWAY_KEY`
   - a volume on `/app/state`

   Before any number from the device goes into BENCHMARKS.md, measure it there with `bench/edge_throughput.py` (point it at the device's agent).
7. **Mosquitto.** On a real network, add a password file and TLS (`deploy/mosquitto/mosquitto.conf` is anonymous, for a private network only).
8. **Corpus licences.** Check each source's reuse terms and record them in `src/aqedge/rag/corpus.py` (`licence_checked=True`) before publishing any chunk.
