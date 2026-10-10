"""Settings shared by the app, the edge gateway and the scripts. Every value
can be overridden with an environment variable."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("AQEDGE_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("AQEDGE_DATA", ROOT / "data"))
REPLAY_PARQUET = DATA_DIR / "raw" / "dhaka_aqi_hourly.parquet"

# One Postgres (with the pgvector extension) holds readings, anomalies and the RAG corpus.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://aq:aq@localhost:5435/aq")

# Where published edge models and models/manifest.json live (served by the app at /models).
PUBLISHED_MODELS = Path(os.environ.get("PUBLISHED_MODELS", ROOT / "models" / "published"))

# The shared secret gateways send in the X-Gateway-Key header.
GATEWAY_KEY = os.environ.get("GATEWAY_KEY", "local-demo-key")

FRESHNESS_SLA_MINUTES = int(os.environ.get("FRESHNESS_SLA_MINUTES", "120"))
REPLAY_EVERY_MINUTES = float(os.environ.get("REPLAY_EVERY_MINUTES", "15"))
REPLAY_ENABLED = os.environ.get("REPLAY_ENABLED", "true") == "true"

POLLUTANTS = ("pm25", "pm10", "no2", "so2", "o3", "co")  # all in µg/m³
