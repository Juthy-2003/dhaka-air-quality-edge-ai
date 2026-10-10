# 0004 — Compare each sensor with its neighbours

**Status:** Accepted (2026-10-10)

## Context
The first drift rule (fast vs slow moving average of one sensor) fired 7,543 times outside faults against 1,154 true hits: a real pollution episode and a drifting sensor look the same from inside one series.

## Decision
Features include `peer_dev`: the sensor's log-ratio to the median of the other sensors measuring the same pollutant at the same time, minus that ratio's recent mean (and its 6-reading average). Real episodes move all four Dhaka locations together; a drifting sensor diverges. The gateway sees every sensor on its site, so it can compute this.

## Consequences
- ✅ The rule baseline's precision rose (0.20 → 0.33); the models gained spike/stuck accuracy.
- ⚠️ Drift recall is still < 20% for the models. A hybrid (model OR peer-drift rule) tuned on 2024 did not generalise to 2025. Recorded as the main open problem.
- Training uses only 2022-08 onward, when all four locations report.
