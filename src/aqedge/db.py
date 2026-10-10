"""Readings and anomalies in Postgres — the same database the RAG corpus uses.

readings   one row per (location, hour). The primary key (location_id, ts)
           makes saving idempotent: `ON CONFLICT DO NOTHING` skips a row
           that is already there, so re-running the replay inserts nothing.
anomalies  events forwarded by edge gateways. The primary key event_id makes
           a resend (after a lost acknowledgement) harmless.

Plain SQL with psycopg; one short-lived connection per call keeps it simple
(the traffic here is a handful of requests a minute).
"""

from datetime import UTC, datetime

import psycopg
from psycopg.rows import dict_row

from aqedge import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    location_id text NOT NULL,
    location text NOT NULL,
    ts timestamptz NOT NULL,
    pm25 real, pm10 real, no2 real, so2 real, o3 real, co real,   -- µg/m³
    source text NOT NULL,              -- 'replay-mendeley' until a live source exists
    replayed_from timestamptz,         -- original time of a replayed reading
    PRIMARY KEY (location_id, ts)
);
CREATE TABLE IF NOT EXISTS anomalies (
    event_id text PRIMARY KEY,
    gateway_id text NOT NULL,
    sensor_id text NOT NULL,
    location text NOT NULL DEFAULT '',
    parameter text NOT NULL,
    ts timestamptz NOT NULL,
    value real,
    score real NOT NULL,
    threshold real NOT NULL,
    model_version text NOT NULL,
    "window" real[],
    sent_at timestamptz NOT NULL,
    queued_for_s real NOT NULL DEFAULT 0,
    received_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS anomalies_by_ts ON anomalies (ts DESC);
"""

READING_COLUMNS = ["location_id", "location", "ts", *config.POLLUTANTS, "source", "replayed_from"]
ANOMALY_COLUMNS = [
    "event_id",
    "gateway_id",
    "sensor_id",
    "location",
    "parameter",
    "ts",
    "value",
    "score",
    "threshold",
    "model_version",
    "window",
    "sent_at",
    "queued_for_s",
]


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(url or config.DATABASE_URL, autocommit=True, row_factory=dict_row)


def init() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)


def save_readings(rows: list[dict]) -> int:
    """Insert rows that are not stored yet; return how many were new."""
    cols = ", ".join(READING_COLUMNS)
    marks = ", ".join(["%s"] * len(READING_COLUMNS))
    sql = f"INSERT INTO readings ({cols}) VALUES ({marks}) ON CONFLICT (location_id, ts) DO NOTHING"
    inserted = 0
    with connect() as conn, conn.cursor() as cur:
        for row in rows:
            cur.execute(sql, [row.get(c) for c in READING_COLUMNS])
            inserted += cur.rowcount
    return inserted


def locations() -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT location_id, location, source, max(ts) AS latest_ts "
            "FROM readings GROUP BY location_id, location, source ORDER BY location"
        ).fetchall()


def latest(location_id: str | None = None, location: str | None = None) -> dict | None:
    """Newest reading for one location (by id or by name), with its age."""
    where, arg = ("location_id = %s", location_id) if location_id else ("location = %s", location)
    with connect() as conn:
        row = conn.execute(
            f"SELECT * FROM readings WHERE {where} AND ts <= now() ORDER BY ts DESC LIMIT 1",
            (arg,),
        ).fetchone()
    if row is None:
        return None
    age = (datetime.now(UTC) - row["ts"]).total_seconds() / 60
    return {**row, "unit": "µg/m³", "age_minutes": round(age, 1), "stale": age > config.FRESHNESS_SLA_MINUTES}


def newest_age_minutes() -> float | None:
    with connect() as conn:
        row = conn.execute("SELECT max(ts) AS ts FROM readings WHERE ts <= now()").fetchone()
    if row["ts"] is None:
        return None
    return round((datetime.now(UTC) - row["ts"]).total_seconds() / 60, 1)


def save_anomaly(event: dict) -> bool:
    """Store one anomaly; return False if this event_id was already stored."""
    cols = ", ".join(f'"{c}"' for c in ANOMALY_COLUMNS)
    marks = ", ".join(["%s"] * len(ANOMALY_COLUMNS))
    with connect() as conn:
        row = conn.execute(
            f"INSERT INTO anomalies ({cols}) VALUES ({marks}) ON CONFLICT (event_id) DO NOTHING RETURNING event_id",
            [event.get(c) for c in ANOMALY_COLUMNS],
        ).fetchone()
    return row is not None


def list_anomalies(location: str | None = None, limit: int = 100) -> list[dict]:
    limit = max(1, min(limit, 5000))
    with connect() as conn:
        if location:
            return conn.execute(
                "SELECT * FROM anomalies WHERE location = %s ORDER BY ts DESC LIMIT %s", (location, limit)
            ).fetchall()
        return conn.execute("SELECT * FROM anomalies ORDER BY ts DESC LIMIT %s", (limit,)).fetchall()
