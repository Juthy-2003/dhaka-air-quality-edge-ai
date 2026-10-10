"""Edge agent: MQTT in, anomalies out.

Subscribes to `aq/<site>/<sensor>/<parameter>` (QoS 1). For every reading it
updates that sensor's window, computes features (including the deviation
from neighbouring sensors of the same pollutant), scores them with the ONNX
model, and — only if anomalous — queues an event for the read API. A null
reading is a dropout and is always flagged.

Bandwidth: it counts the bytes it actually forwards and the bytes it would
have sent by forwarding every reading, so `1 − forwarded/all` is measured,
not assumed. Latency: if a message carries `published_at` (the simulator's clock),
the agent records how long it took from publish to "scored". Both go to
state/stats.json every 5 s; Prometheus metrics on :9200.

    python -m aqedge.edge.agent --mqtt localhost --api http://localhost:8000 --model-dir models/current
"""

import argparse
import collections
import json
import logging
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import requests
from prometheus_client import Counter, Gauge, start_http_server

from aqedge.edge.features import WINDOW, features, peer_log_ratio
from aqedge.edge.queue import Outbox
from aqedge.edge.scorer import Scorer

log = logging.getLogger("edge.agent")

READINGS = Counter("edge_readings_total", "Readings received")
DUPLICATES = Counter("edge_duplicate_readings_total", "QoS-1 redeliveries dropped")
ANOMALIES = Counter("edge_anomalies_total", "Anomalies detected", ["kind"])
BYTES_ALL = Counter("edge_bytes_if_forward_all_total", "Bytes if every reading were forwarded")
BYTES_SENT = Counter("edge_bytes_forwarded_total", "Bytes actually forwarded upstream")
QUEUE = Gauge("edge_outbox_depth", "Anomalies waiting for the uplink")
MODEL = Gauge("edge_model_info", "Model in use (value 1)", ["version"])
SCORE_SECONDS = Counter("edge_score_seconds_total", "Time spent in model inference")


class Agent:
    def __init__(self, model_dir: Path, api: str, gateway_id: str, key: str, outbox: Path):
        self.model_dir = Path(model_dir)
        self.scorer = Scorer(self.model_dir)
        self._set_model_metric()
        self.api, self.gateway_id, self.key = api.rstrip("/"), gateway_id, key
        self.outbox = Outbox(outbox)
        self.values: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=WINDOW))
        self.peer_hist: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=WINDOW))
        self.latest: dict[tuple[str, str], tuple[str, float]] = {}  # (parameter, sensor) → (ts, value)
        self.seen: collections.OrderedDict = collections.OrderedDict()
        self.lock = threading.Lock()
        self.stats = {"readings": 0, "anomalies": 0, "bytes_all": 0, "bytes_sent": 0}
        self.latency_ms = collections.deque(maxlen=100_000)  # publish → scored (last 100k messages with published_at)
        self.first_at = self.last_at = None  # wall-clock time of the first / last message
        self.score_seconds = 0.0
        self.scored = 0

    def _set_model_metric(self):
        MODEL.clear()
        MODEL.labels(self.scorer.version).set(1)

    def reload_model(self) -> str:
        """Called by the OTA updater after it has swapped files atomically."""
        with self.lock:
            self.scorer = Scorer(self.model_dir)
            self._set_model_metric()
        return self.scorer.version

    def _dedupe(self, sensor: str, seq: int) -> bool:
        key = (sensor, seq)
        if key in self.seen:
            return True
        self.seen[key] = None
        if len(self.seen) > 20000:
            self.seen.popitem(last=False)
        return False

    def handle(self, payload: bytes) -> dict | None:
        msg = json.loads(payload)
        try:
            return self._handle(msg, payload)
        finally:
            now = time.time()
            self.first_at = self.first_at or now
            self.last_at = now
            if "published_at" in msg:
                self.latency_ms.append((now - msg["published_at"]) * 1000)

    def _handle(self, msg: dict, payload: bytes) -> dict | None:
        READINGS.inc()
        self.stats["readings"] += 1
        BYTES_ALL.inc(len(payload))
        self.stats["bytes_all"] += len(payload)
        sensor, param = msg["sensor_id"], msg["parameter"]
        if self._dedupe(sensor, msg["seq"]):
            DUPLICATES.inc()
            return None
        x = np.nan if msg["value"] is None else float(msg["value"])
        self.latest[(param, sensor)] = (msg["ts"], x)
        peers = np.array(
            [v for (p, s), (ts, v) in self.latest.items() if p == param and s != sensor and ts == msg["ts"]]
        )
        ratio = peer_log_ratio(x, peers) if len(peers) else np.nan
        hist = np.array(self.values[sensor], dtype=float)
        phist = np.array(self.peer_hist[sensor], dtype=float)
        feats = features(hist, x, phist, ratio)
        self.values[sensor].append(x)
        self.peer_hist[sensor].append(ratio)

        kind, score = None, None
        if np.isnan(x):
            kind = "dropout"
        elif not np.isnan(feats).any():
            t = time.perf_counter()
            with self.lock:
                score = self.scorer.score(feats)
                threshold = self.scorer.threshold
                version = self.scorer.version
            dt = time.perf_counter() - t
            SCORE_SECONDS.inc(dt)
            self.score_seconds += dt
            self.scored += 1
            if score > threshold:
                kind = "model"
        if kind is None:
            return None
        event = {
            "event_id": uuid.uuid5(uuid.NAMESPACE_URL, f"{self.gateway_id}/{sensor}/{msg['seq']}").hex,
            "gateway_id": self.gateway_id,
            "sensor_id": sensor,
            "location": msg.get("location", ""),
            "parameter": param,
            "ts": msg["ts"],
            "value": None if np.isnan(x) else x,
            "score": score if score is not None else 1.0,
            "threshold": self.scorer.threshold,
            "model_version": "rule:dropout" if kind == "dropout" else version,
            "window": [None if np.isnan(v) else float(v) for v in hist[-6:]],
        }
        self.outbox.put(event)
        QUEUE.set(self.outbox.depth())
        ANOMALIES.labels(kind).inc()
        self.stats["anomalies"] += 1
        return event

    def summary(self) -> dict:
        """What state/stats.json holds: counters, model, latency and throughput."""
        lat = np.array(self.latency_ms) if self.latency_ms else None
        span = (self.last_at - self.first_at) if self.first_at else 0
        return {
            **self.stats,
            "queue": self.outbox.depth(),
            "model": self.scorer.version,
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 2) if lat is not None else None,
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2) if lat is not None else None,
            "received_per_s": round(self.stats["readings"] / span, 1) if span > 0 else None,
            "score_ms_mean": round(self.score_seconds / self.scored * 1000, 3) if self.scored else None,
        }

    def send_loop(self, stop: threading.Event, interval: float = 1.0) -> None:
        """Drain the outbox while the uplink works; back off when it does not."""
        backoff = interval
        session = requests.Session()
        session.headers.update({"X-Gateway-Key": self.key, "Content-Type": "application/json"})
        while not stop.is_set():
            batch = self.outbox.peek()
            if not batch:
                stop.wait(interval)
                continue
            try:
                for row_id, event, queued_at in batch:
                    body = {
                        **event,
                        "sent_at": datetime.now(UTC).isoformat(),
                        "queued_for_s": round(time.time() - queued_at, 2),
                    }
                    data = json.dumps(body)
                    r = session.post(f"{self.api}/anomalies", data=data, timeout=5)
                    if r.status_code in (200, 201):
                        BYTES_SENT.inc(len(data))
                        self.stats["bytes_sent"] += len(data)
                        self.outbox.ack(row_id)
                    else:
                        raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
                backoff = interval
            except requests.RequestException as exc:
                log.warning(
                    "uplink down (%s); %d queued, retry in %.0fs", type(exc).__name__, self.outbox.depth(), backoff
                )
                stop.wait(backoff)
                backoff = min(backoff * 2, 30)
            QUEUE.set(self.outbox.depth())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqtt", default=os.environ.get("MQTT_HOST", "localhost"))
    ap.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/current"))
    ap.add_argument("--gateway-id", default=os.environ.get("GATEWAY_ID", "gw-dhaka-01"))
    ap.add_argument("--key", default=os.environ.get("GATEWAY_KEY", "local-demo-key"))
    ap.add_argument("--outbox", default=os.environ.get("OUTBOX", "state/outbox.sqlite"))
    ap.add_argument("--stats", default=os.environ.get("STATS_FILE", "state/stats.json"))
    a = ap.parse_args()
    start_http_server(int(os.environ.get("METRICS_PORT", "9200")))
    model_dir = Path(a.model_dir)
    if not (model_dir / "meta.json").exists():
        # First boot: copy the model baked into the image onto writable storage.
        # OTA swaps directories with rename(), which needs a normal filesystem
        # (a directory inside the image's read-only layers cannot be renamed).
        import shutil

        shutil.copytree(os.environ.get("BUNDLED_MODEL_DIR", "/app/models/current"), model_dir)
    agent = Agent(model_dir, a.api, a.gateway_id, a.key, Path(a.outbox))
    stop = threading.Event()
    threading.Thread(target=agent.send_loop, args=(stop,), daemon=True).start()
    if os.environ.get("OTA_ENABLED", "true") == "true":
        from aqedge.edge.ota import Updater

        threading.Thread(target=Updater(model_dir, agent.reload_model).loop, args=(stop,), daemon=True).start()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"agent-{a.gateway_id}", clean_session=False)
    client.on_connect = lambda c, *_: c.subscribe("aq/+/+/+", qos=1)
    client.on_message = lambda c, u, m: agent.handle(m.payload)
    client.connect(a.mqtt, 1883)
    client.loop_start()
    try:
        while True:
            time.sleep(5)
            Path(a.stats).write_text(json.dumps(agent.summary()))
    finally:
        stop.set()
        client.loop_stop()


if __name__ == "__main__":
    main()
