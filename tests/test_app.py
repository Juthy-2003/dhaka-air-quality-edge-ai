"""Readings idempotency and the app's endpoints, against a real Postgres (conftest.py)."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient


def reading(hours_ago, pm25, loc="L1", name="Mirpur"):
    ts = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours_ago)
    return {
        "location_id": loc,
        "location": name,
        "ts": ts,
        "pm25": pm25,
        "pm10": pm25 * 2,
        "source": "replay-mendeley",
        "replayed_from": ts - timedelta(days=730),
    }


def test_saving_readings_is_idempotent(database):
    batch = [reading(3, 50), reading(2, 60), reading(1, 70)]
    assert database.save_readings(batch) == 3
    assert database.save_readings(batch) == 0  # the same replay run again
    assert database.save_readings(batch + [reading(0, 80)]) == 1
    assert database.latest(location_id="L1")["pm25"] == 80


def test_replay_as_live_end_to_end(database):
    from aqedge.app import replay_once
    from aqedge.data import replay

    df = replay.load()
    now = datetime(2026, 3, 1, 12, tzinfo=UTC)
    first, again = replay_once(df, now), replay_once(df, now)
    assert first == 4 * 48 and again == 0  # 4 Dhaka locations × 48 hours, then nothing new
    row = database.latest(location="Paltan")
    assert row["source"] == "replay-mendeley" and row["ts"] - row["replayed_from"] == replay.REPLAY_LAG


@pytest.fixture
def client(database):
    from aqedge.app import app

    database.save_readings([reading(2, 40, loc="L9", name="Tungi"), reading(1, 45, loc="L9", name="Tungi")])
    with TestClient(app) as c:
        yield c


def test_locations_and_latest(client):
    assert any(x["location_id"] == "L9" for x in client.get("/locations").json())
    latest = client.get("/locations/L9/latest").json()
    assert latest["pm25"] == 45 and latest["source"] == "replay-mendeley"
    assert 0 <= latest["age_minutes"] < 120 and latest["stale"] is False
    assert client.get("/locations/nope/latest").status_code == 404
    assert client.get("/health").json()["ok"] is True


def test_anomaly_post_needs_key_and_is_idempotent(client):
    event = {
        "event_id": "e1",
        "gateway_id": "gw-test",
        "sensor_id": "tungi-pm25",
        "location": "Tungi",
        "parameter": "pm25",
        "ts": "2026-10-10T08:00:00Z",
        "value": 400,
        "score": 0.9,
        "threshold": 0.5,
        "model_version": "3",
        "window": [40, None, 41],
        "sent_at": "2026-10-10T08:00:02Z",
    }
    assert client.post("/anomalies", json=event).status_code == 401
    key = {"X-Gateway-Key": "local-demo-key"}
    assert client.post("/anomalies", json=event, headers=key).status_code == 201
    second = client.post("/anomalies", json=event, headers=key)
    assert second.status_code == 200 and second.json()["duplicate"] is True
    stored = client.get("/anomalies", params={"location": "Tungi"}).json()
    assert len(stored) == 1 and stored[0]["window"] == [40, None, 41]


def test_place_resolution_and_category():
    from aqedge.rag.aqi import pm25_category
    from aqedge.rag.locations import resolve

    assert resolve("Is it safe to jog in Mirpur this evening?") == ("Mirpur", "Azimpur")
    assert resolve("What is the NowCast?") is None
    assert pm25_category(44.0) == "Unhealthy for Sensitive Groups" and pm25_category(9.0) == "Good"
