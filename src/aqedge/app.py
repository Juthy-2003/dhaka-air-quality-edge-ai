"""The one backend app: readings, anomalies, the RAG assistant and the model manifest.

    uvicorn aqedge.app:app --port 8000          # OpenAPI docs at /docs

GET  /health                    newest reading's age
GET  /locations                 monitored locations
GET  /locations/{id}/latest     newest reading, with age_minutes and stale
POST /anomalies                 edge gateways forward anomalies (X-Gateway-Key)
GET  /anomalies                 what the gateways forwarded
POST /ask                       grounded answer (rag/service.py)
GET  /models/manifest.json      the edge model the gateways should run (OTA)
GET  /metrics                   Prometheus metrics

On startup a background thread replays historical data as live every
REPLAY_EVERY_MINUTES (same hour, 730 days earlier, labelled replay-mendeley).
"""

import hmac
import logging
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from aqedge import __version__, config, db
from aqedge.rag import service

log = logging.getLogger("aqedge.app")


def replay_once(df, now: datetime | None = None) -> int:
    """Store the last 48 replayed hours; already-stored rows are skipped."""
    from aqedge.data import replay

    end = (now or datetime.now(UTC)).replace(minute=0, second=0, microsecond=0)
    return db.save_readings(replay.replay_as_live(df, end - timedelta(hours=48), end))


def replay_loop(stop: threading.Event) -> None:
    from aqedge.data import replay

    df = replay.load()
    while not stop.is_set():
        try:
            log.warning("replay: %d new readings", replay_once(df))
        except Exception:  # keep going; the next run catches up
            log.exception("replay failed")
        stop.wait(config.REPLAY_EVERY_MINUTES * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    stop = threading.Event()
    if config.REPLAY_ENABLED:
        threading.Thread(target=replay_loop, args=(stop,), daemon=True).start()
    yield
    stop.set()


app = FastAPI(
    title="Dhaka Air Quality — Edge AI + RAG",
    version=__version__,
    description="Readings are historical data replayed as live (`source: replay-mendeley`). "
    "The assistant is a demo, not medical advice.",
    lifespan=lifespan,
)
config.PUBLISHED_MODELS.mkdir(parents=True, exist_ok=True)
app.mount("/models", StaticFiles(directory=config.PUBLISHED_MODELS), name="models")


class AnomalyIn(BaseModel):
    event_id: str
    gateway_id: str
    sensor_id: str
    location: str = ""
    parameter: str
    ts: datetime
    value: float | None = None
    score: float
    threshold: float
    model_version: str
    window: list[float | None] = []
    sent_at: datetime
    queued_for_s: float = 0


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True, "newest_reading_age_min": db.newest_age_minutes(), "version": __version__}


@app.get("/locations")
def locations():
    return db.locations()


@app.get("/locations/{location_id}/latest")
def latest(location_id: str):
    row = db.latest(location_id=location_id)
    if row is None:
        raise HTTPException(404, "unknown location")
    return row


@app.post("/anomalies", status_code=201)
def post_anomaly(body: AnomalyIn, response: Response, x_gateway_key: str = Header(default="")):
    """Idempotent on event_id: a resend returns 200 with duplicate=true."""
    if not hmac.compare_digest(x_gateway_key, config.GATEWAY_KEY):
        raise HTTPException(401, "bad gateway key")
    if db.save_anomaly(body.model_dump()):
        return {"accepted": True, "duplicate": False}
    response.status_code = 200
    return {"accepted": True, "duplicate": True}


@app.get("/anomalies")
def anomalies(location: str | None = None, limit: int = 100):
    return db.list_anomalies(location, limit)


@app.post("/ask")
def ask(body: Question):
    return service.ask(body.question)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
