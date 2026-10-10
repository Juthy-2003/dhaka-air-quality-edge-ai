# 0003 — Ship the isolation forest; INT8 only where it applies

**Status:** Accepted (2026-10-10)

> **Update (2026-10-10, trim, [0009](0009-trim-for-defensibility.md)):** the autoencoder and the INT8 code were removed. The INT8 finding below is kept as history: it was measured before the trim and was not re-measured.

## Decision
The gateway runs the isolation forest (ONNX, 1.74 MB): best F1 on the 2025 test year (0.82; precision 0.90). The autoencoder (F1 0.76) stays registered as an alternative and is the INT8 case study.

## Why INT8 is not the headline
Dynamic INT8 quantisation targets MatMul/Conv operators. The isolation forest has none, so INT8 does nothing for it. For the tiny autoencoder (~300 weights) INT8 made the file larger (2.6 → 4.8 KB) and inference slower (8.7 → 11.6 µs on the Mac; 5.5 → 8.3 µs in the gateway container) because quantise/dequantise overhead exceeds the savings. F1 barely moved (0.762 → 0.759). The plan predicted exactly this possibility; the measured answer is "INT8 is for large matrix-heavy models".

## Consequences
- ✅ 0.8–0.9 ms per reading on half a CPU, ≈ 85 MB process: far inside a gateway's budget for 8 readings an hour.
- ⚠️ Drift recall stays low (ADR 0004).
