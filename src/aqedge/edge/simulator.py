"""SYNTHETIC sensor stream for the gateway.

Replays historical Dhaka readings (Mendeley, modelled values — see
data/replay.py) as virtual sensors over MQTT, injects faults at known
positions, and writes the ground truth to a labels file the agent never
sees. One message per sensor per simulated hour, published in timestamp
order so the agent can compare each sensor with its neighbours.

Each message carries `published_at` (wall clock) so the agent can measure
end-to-end latency; `--rate` paces publishing, which is how the throughput
benchmark (bench/edge_throughput.py) offers more and more load.

    python -m aqedge.edge.simulator --mqtt localhost --start 2025-01-01 --hours 2000 --rate 400
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import pandas as pd

from aqedge.data import replay
from aqedge.edge import faults as F

LOCATIONS = ("Dhaka", "Azimpur", "Paltan", "Tungi")
PARAMS = ("pm25", "pm10")


def build(start: str, hours: int, seed: int, faults_per_sensor: int):
    df = replay.load()
    streams, labels = {}, []
    idx = None
    for p_i, param in enumerate(PARAMS):
        m = pd.DataFrame({loc: replay.series(df, loc, param) for loc in LOCATIONS}).loc[start:].iloc[:hours]
        idx = m.index
        for l_i, loc in enumerate(LOCATIONS):
            vals, fl = F.inject(m[loc].to_numpy(dtype=float), faults_per_sensor, seed=seed * 100 + p_i * 10 + l_i)
            sensor = f"{loc.lower()}-{param}"
            streams[(sensor, loc, param)] = vals
            labels += [
                {
                    "sensor_id": sensor,
                    "kind": f.kind,
                    "start_ts": str(idx[f.start]),
                    "end_ts": str(idx[min(f.end, len(idx) - 1)]),
                    "start": f.start,
                    "end": f.end,
                }
                for f in fl
            ]
    return idx, streams, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqtt", default=os.environ.get("MQTT_HOST", "localhost"))
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--hours", type=int, default=2000)
    ap.add_argument("--rate", type=float, default=400, help="messages per second")
    ap.add_argument("--faults", type=int, default=6, help="faults per sensor")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--labels", default="state/labels.json")
    a = ap.parse_args()
    idx, streams, labels = build(a.start, a.hours, a.seed, a.faults)
    Path(a.labels).parent.mkdir(parents=True, exist_ok=True)
    Path(a.labels).write_text(json.dumps({"synthetic_faults": True, "labels": labels}, indent=1))
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="simulator")
    client.max_inflight_messages_set(1000)  # QoS 1: up to 1000 messages awaiting the broker's PUBACK
    client.connect(a.mqtt, 1883)
    client.loop_start()
    sent, t0, pending = 0, time.perf_counter(), []
    for i, ts in enumerate(idx):
        for (sensor, loc, param), vals in streams.items():
            v = vals[i]
            msg = {
                "seq": i,
                "ts": ts.isoformat(),
                "sensor_id": sensor,
                "location": loc,
                "parameter": param,
                "value": None if np.isnan(v) else round(float(v), 2),
                "synthetic": True,
                "published_at": time.time(),
            }
            pending.append(client.publish(f"aq/dhaka/{sensor}/{param}", json.dumps(msg), qos=1))
            sent += 1
            ahead = sent / a.rate - (time.perf_counter() - t0)
            if ahead > 0:
                time.sleep(ahead)
    for info in pending:  # wait until the broker has acknowledged every message (PUBACK)
        info.wait_for_publish()
    client.loop_stop()
    seconds = time.perf_counter() - t0
    print(
        json.dumps(
            {"published": sent, "seconds": round(seconds, 1), "rate": round(sent / seconds, 1), "faults": len(labels)}
        )
    )


if __name__ == "__main__":
    main()
