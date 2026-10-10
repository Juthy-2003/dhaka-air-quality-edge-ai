"""Score what the gateway actually forwarded against the simulator's labels.

Reads anomalies from the app and the ground truth from state/labels.json.
A fault is detected if the gateway forwarded an event for that sensor inside
the fault or within 2 hours after it; an event episode (consecutive hours on
one sensor) is a true positive if it overlaps a fault ±2 hours. Bandwidth
comes from the agent's own byte counters (state/stats.json).

    uv run python scripts/edge_report.py [http://localhost:8000]
"""

import json
import sys
from collections import defaultdict

import pandas as pd
import requests

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
TOL = pd.Timedelta(hours=2)


def utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


labels = json.load(open("state/labels.json"))["labels"]
stats = json.load(open("state/stats.json"))
events = requests.get(f"{API}/anomalies", params={"limit": 5000}, timeout=10).json()
by_sensor = defaultdict(list)
for e in events:
    by_sensor[e["sensor_id"]].append(utc(e["ts"]))
faults = defaultdict(list)
for lab in labels:
    faults[lab["sensor_id"]].append((lab["kind"], utc(lab["start_ts"]), utc(lab["end_ts"])))

detected = defaultdict(lambda: [0, 0])  # kind → [detected, total]
for sensor, fl in faults.items():
    ts = by_sensor.get(sensor, [])
    for kind, s, e in fl:
        detected[kind][1] += 1
        detected[kind][0] += int(any(s <= t <= e + TOL for t in ts))
tp = fp = 0
for sensor, ts in by_sensor.items():
    ts = sorted(ts)
    episodes, cur = [], [ts[0]]
    for t in ts[1:]:
        if t - cur[-1] <= pd.Timedelta(hours=1):
            cur.append(t)
        else:
            episodes.append(cur)
            cur = [t]
    episodes.append(cur)
    for ep in episodes:
        hit = any(ep[0] <= e + TOL and ep[-1] >= s - TOL for _, s, e in faults.get(sensor, []))
        tp, fp = tp + hit, fp + (not hit)
total_det = sum(v[0] for v in detected.values())
total = sum(v[1] for v in detected.values())
queued = [e["queued_for_s"] for e in events]
print(
    json.dumps(
        {
            "readings_received": stats["readings"],
            "events_forwarded": len(events),
            "bandwidth_saved": round(1 - stats["bytes_sent"] / stats["bytes_all"], 4) if stats["bytes_all"] else None,
            "bytes_all_vs_sent": [stats["bytes_all"], stats["bytes_sent"]],
            "recall": round(total_det / total, 4),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall_by_kind": {k: f"{v[0]}/{v[1]}" for k, v in detected.items()},
            "events_by_model_version": pd.Series([e["model_version"] for e in events]).value_counts().to_dict(),
            "max_queued_for_s": max(queued) if queued else None,
            "events_that_waited_in_outbox": sum(q > 2 for q in queued),
            "outbox_left": stats["queue"],
            "model_running_at_end": stats["model"],
        },
        indent=2,
    )
)
