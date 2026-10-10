.PHONY: setup data train test lint up demo llm-model ingest eval answers bench publish down

setup:      ## install (incl. the training and dev groups)
	uv sync
data:       ## Dhaka replay data (downloads 303 MB once, keeps ~17 MB)
	uv run python scripts/extract_dhaka.py
train:      ## isolation forest → models/current, bench/results/anomaly.json
	uv run python scripts/train_anomaly.py
test:       ## needs Postgres: starts the compose one on :5435
	docker compose -p aq up -d postgres --wait
	uv run pytest -q
lint:
	uv run ruff check . && uv run ruff format --check .
up:         ## postgres, app :8000 (/docs), llm, mosquitto, edge agent (0.5 CPU / 256 MB)
	mkdir -p models/published state
	docker compose -p aq up -d --build --wait
	docker builder prune -f > /dev/null
demo:       ## SYNTHETIC sensors, uplink outage, OTA swap mid-stream → report
	./scripts/edge_demo.sh
llm-model:  ## Qwen2.5-1.5B-Instruct Q4_K_M (1.1 GB, Apache-2.0)
	mkdir -p data/llm && curl -L -o data/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf \
	  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
ingest:     ## corpus → chunks → embeddings → pgvector
	uv run python -m aqedge.rag.ingest
eval:       ## retrieval gate (recall@5, MRR vs baseline)
	uv run python -m aqedge.rag.evaluate --gate
answers:    ## answer-level check of the running assistant
	uv run python bench/rag_answers.py
bench:      ## edge throughput: simulator → Mosquitto → agent (0.5 CPU / 256 MB)
	uv run python bench/edge_throughput.py
publish:    ## publish models/current as the next version in models/published/manifest.json
	uv run python scripts/publish_model.py
down:
	docker compose -p aq down
