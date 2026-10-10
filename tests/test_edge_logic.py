import json

import numpy as np
import pytest

from aqedge.edge import faults as F
from aqedge.edge.features import FEATURES, WINDOW, features, peer_log_ratio, series_features
from aqedge.edge.queue import Outbox
from aqedge.edge.rules import rule_flags


def smooth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 60 + 20 * np.sin(t / 24 * 2 * np.pi) + rng.normal(0, 2, n)


def test_series_features_match_streaming_features():
    """Training (series_features) and the gateway (features) must agree exactly."""
    v, r = smooth(), np.zeros(400)
    batch = series_features(v, r)
    i = 200
    one = features(v[i - WINDOW : i], v[i], r[i - WINDOW : i], r[i])
    np.testing.assert_allclose(batch[i], one)
    assert batch.shape == (400, len(FEATURES))


def test_feature_signatures_of_faults():
    v, r = smooth(), np.zeros(400)
    stuck = features(np.append(v[150:195], [v[194]] * 3), v[194], r[150:198], 0.0)
    assert stuck[FEATURES.index("flat_len")] >= 3
    spike = features(v[150:198], v[198] * 5, r[150:198], 0.0)
    assert spike[FEATURES.index("log_ratio")] > 1
    assert np.isnan(features(v[150:198], np.nan, r[150:198], 0.0)).all()


def test_peer_ratio_neutral_without_neighbours():
    assert np.isnan(peer_log_ratio(50, np.array([np.nan])))
    f = features(smooth()[:48], 60.0, np.full(48, np.nan), np.nan)
    assert f[FEATURES.index("peer_dev")] == 0.0


def test_inject_keeps_faults_inside_and_labels_them():
    v = smooth(3000)
    out, fl = F.inject(v, 40, seed=1)
    assert all(0 < f.start < f.end <= len(v) for f in fl)
    assert {f.kind for f in fl} == set(F.KINDS)
    y = F.labels(len(v), fl)
    assert y.sum() == sum(f.end - f.start for f in fl)
    drop = next(f for f in fl if f.kind == "dropout")
    assert np.isnan(out[drop.start : drop.end]).all()


def test_rules():
    f = np.zeros(len(FEATURES), dtype=np.float32)
    assert rule_flags(f, np.nan) == (True, "dropout")
    assert rule_flags(f, 50.0) == (False, "")
    f[FEATURES.index("flat_len")] = 5
    assert rule_flags(f, 50.0)[1] == "stuck"


def test_outbox_survives_restart_and_dedupes(tmp_path):
    ob = Outbox(tmp_path / "o.sqlite")
    ob.put({"event_id": "a", "x": 1})
    ob.put({"event_id": "a", "x": 1})  # same event twice (QoS-1 redelivery)
    ob.put({"event_id": "b", "x": 2})
    assert ob.depth() == 2
    ob2 = Outbox(tmp_path / "o.sqlite")  # "reboot"
    rows = ob2.peek()
    assert [e["event_id"] for _, e, _ in rows] == ["a", "b"]
    ob2.ack(rows[0][0])
    assert ob2.depth() == 1


@pytest.fixture
def agent(tmp_path):
    from pathlib import Path

    from aqedge.edge.agent import Agent

    return Agent(Path("models/current"), "http://api.invalid", "gw-test", "k", tmp_path / "outbox.sqlite")


def msg(seq, sensor, value, ts, param="pm25"):
    return json.dumps(
        {"seq": seq, "ts": ts, "sensor_id": sensor, "location": sensor, "parameter": param, "value": value}
    ).encode()


def test_agent_forwards_only_anomalies(agent):
    v = smooth(200)
    for i in range(120):
        ts = f"2025-01-{1 + i // 24:02d}T{i % 24:02d}:00:00"
        for s, k in (("a", 1.0), ("b", 1.05), ("c", 0.95)):
            agent.handle(msg(i, s, round(float(v[i] * k), 2), ts))
    quiet = agent.stats["anomalies"]
    assert quiet <= 3  # clean data: (almost) nothing forwarded
    ts = "2025-01-06T00:00:00"
    agent.handle(msg(120, "b", round(float(v[120]), 2), ts))
    agent.handle(msg(120, "c", round(float(v[120]), 2), ts))
    spike = agent.handle(msg(120, "a", float(v[120] * 6), ts))
    assert spike is not None and spike["model_version"] == "local"
    drop = agent.handle(msg(121, "a", None, "2025-01-06T01:00:00"))
    assert drop["model_version"] == "rule:dropout"
    assert agent.handle(msg(121, "a", None, "2025-01-06T01:00:00")) is None  # QoS-1 duplicate
    assert agent.stats["bytes_all"] > 0 and agent.outbox.depth() == agent.stats["anomalies"]
    assert spike["gateway_id"] == "gw-test" and agent.summary()["received_per_s"] is not None


def test_chunker_never_duplicates_or_cuts_words():
    from aqedge.rag.chunk import chunk_pages
    from aqedge.rag.corpus import Page

    lines = [f"Line {i} says something about air quality and health guidance." for i in range(60)]
    page = Page("doc", 1, "\n".join(lines))  # no blank lines, like many PDF pages
    chunks = chunk_pages([page], size=300, overlap=80)
    assert all(len(c.text) <= 300 for c in chunks)
    for c in chunks:
        assert len(set(c.text.split("\n"))) == len(c.text.split("\n"))  # no repeated line inside a chunk
        assert all(line in lines for line in c.text.split("\n"))  # every line whole, never cut
    covered = {line for c in chunks for line in c.text.split("\n")}
    assert covered == set(lines)  # nothing dropped
