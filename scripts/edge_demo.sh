#!/usr/bin/env bash
# End-to-end gateway demo: SYNTHETIC sensors → MQTT → agent → app, with an
# uplink outage and an over-the-air model swap while the stream is running.
# Needs `make up` first. Takes about 2 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${PROJECT:-aq}
dc() { docker compose -p "$P" "$@"; }

echo "reset: empty anomalies table, fresh gateway state"
dc exec -T postgres psql -q -U aq -c "TRUNCATE anomalies" > /dev/null
dc stop agent > /dev/null
rm -rf state/model state/model.previous state/outbox.sqlite* state/stats.json state/labels.json
rm -rf models/published/*   # empty it, don't delete it: the app bind-mounts this directory
uv run python scripts/publish_model.py --version 1 > /dev/null          # v1 = models/current
dc start agent > /dev/null
until grep -q '"model": "1"' state/stats.json 2>/dev/null; do sleep 1; done
echo "gateway running model v1 (OTA from the manifest)"

dc run --rm -d --name aq-sim simulator > /dev/null                      # 16,000 readings at 100/s ≈ 160 s
sleep 40  # the first injected faults start ~200 simulated hours in
echo "uplink DOWN"; dc stop app > /dev/null
sleep 20
echo "uplink UP";   dc start app > /dev/null
sleep 3
echo "publish v2";  uv run python scripts/publish_model.py --version 2 > /dev/null
while docker ps --format '{{.Names}}' | grep -q '^aq-sim$'; do sleep 2; done
sleep 15  # let the outbox drain
uv run python scripts/edge_report.py
