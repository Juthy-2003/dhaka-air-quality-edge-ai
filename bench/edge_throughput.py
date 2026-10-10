"""How many sensor messages per second can the gateway keep up with?

Pipeline under test: simulator → Mosquitto → agent (0.5 CPU / 256 MB) →
features + ONNX isolation forest → SQLite outbox → app. The data is
SYNTHETIC (replayed modelled values + injected faults, as in the demo).

For each offered rate the agent is restarted with empty state, the simulator
publishes N messages at that rate (QoS 1), and the agent's own stats
(state/stats.json) give: messages received, achieved rate, end-to-end
latency (simulator publish → agent scored) p50/p95 and mean model time per
reading. `docker stats` is sampled every second for the agent's peak memory
and CPU. A rate is "sustained" if every message arrived and p95 latency
stayed under 1 s.

    uv run python bench/edge_throughput.py           # needs `make up`
"""

import json
import platform
import subprocess
import threading
import time
from pathlib import Path

RATES = [250, 500, 1000, 2000, 4000]
SECONDS_PER_STEP = 20
MAX_HOURS = 7000  # the 2025 replay data has ~7,900 hours; 8 messages per hour
STATS = Path("state/stats.json")
P = ["docker", "compose", "-p", "aq"]


def to_mib(text: str) -> float:
    """'52.3MiB' → 52.3 (docker stats units)."""
    for unit, factor in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024)):
        if text.endswith(unit):
            return float(text[: -len(unit)]) * factor
    return 0.0


def sample_agent(stop: threading.Event, out: list) -> None:
    while not stop.is_set():
        line = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}|{{.CPUPerc}}", "aq-agent-1"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if line:
            mem, cpu = line.split("|")
            out.append((to_mib(mem.split("/")[0].strip()), float(cpu.rstrip("%"))))
        stop.wait(1)


def fresh_agent() -> None:
    subprocess.run([*P, "stop", "agent"], check=True, capture_output=True)
    for f in ["stats.json", "outbox.sqlite", "outbox.sqlite-wal", "outbox.sqlite-shm"]:
        (Path("state") / f).unlink(missing_ok=True)
    subprocess.run([*P, "start", "agent"], check=True, capture_output=True)
    while not STATS.exists():
        time.sleep(0.5)


def step(rate: int) -> dict:
    fresh_agent()
    hours = min(MAX_HOURS, rate * SECONDS_PER_STEP // 8)
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=sample_agent, args=(stop, samples))
    sampler.start()
    sim = subprocess.run(
        [
            *P,
            "run",
            "--rm",
            "simulator",
            "python",
            "-m",
            "aqedge.edge.simulator",
            "--hours",
            str(hours),
            "--rate",
            str(rate),
            "--labels",
            "/state/bench_labels.json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    published = json.loads(sim.stdout.strip().splitlines()[-1])
    # Wait until the agent has stopped receiving (stats are written every 5 s).
    last = -1
    while True:
        time.sleep(6)
        stats = json.loads(STATS.read_text())
        if stats["readings"] in (last, published["published"]):
            break
        last = stats["readings"]
    time.sleep(6)
    stats = json.loads(STATS.read_text())
    stop.set()
    sampler.join()
    lost = published["published"] - stats["readings"]
    return {
        "offered_per_s": rate,
        "published": published["published"],
        "published_per_s": published["rate"],
        "received": stats["readings"],
        "lost": lost,
        "received_per_s": stats["received_per_s"],
        "latency_ms_p50": stats["latency_ms_p50"],
        "latency_ms_p95": stats["latency_ms_p95"],
        "score_ms_mean": stats["score_ms_mean"],
        "anomalies_queued": stats["anomalies"],
        "agent_peak_mem_mib": round(max((m for m, _ in samples), default=0), 1),
        "agent_peak_cpu_pct": max((c for _, c in samples), default=0),
        "sustained": lost == 0 and (stats["latency_ms_p95"] or 1e9) < 1000,
    }


def main() -> None:
    rows = []
    for rate in RATES:
        rows.append(step(rate))
        print(json.dumps(rows[-1]))
    sustained = [r["offered_per_s"] for r in rows if r["sustained"]]
    out = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "data": "SYNTHETIC: replayed modelled Mendeley values for 8 virtual sensors + injected faults",
        "setup": "docker compose on an Apple M4 laptop: simulator (uncapped) → Mosquitto 2.0 → agent container "
        "(--cpus 0.5, 256 MB, ONNX isolation forest, 1 thread) → SQLite outbox → app; MQTT QoS 1",
        "machine": platform.platform(),
        "max_sustained_offered_per_s": max(sustained) if sustained else None,
        "rows": rows,
    }
    Path("bench/results/edge_throughput.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
