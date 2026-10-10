"""Train the edge anomaly model, evaluate it on 2025 with injected faults and
export it to ONNX for the gateway.

    uv run python scripts/train_anomaly.py

Writes models/current/ (iforest.onnx + meta.json, bundled into the gateway
image), src/aqedge/edge/ota_sample.json (the OTA smoke-test sample) and
bench/results/anomaly.json. Publishing a model to running gateways is a
separate step: scripts/publish_model.py.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from aqedge.edge.features import FEATURES, WINDOW
from aqedge.models import anomaly as A

OUT = Path("models/current")
RESULTS = Path("bench/results")
SHOW = [
    "precision",
    "recall",
    "f1",
    "false_alarm_episodes",
    "recall_stuck",
    "recall_spike",
    "recall_drift",
    "recall_dropout",
]


def onnx_scores(path: Path, X: np.ndarray) -> np.ndarray:
    """Scores of the EXPORTED model (higher = more anomalous); NaN rows stay NaN."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    s = np.full(len(X), np.nan)
    ok = ~np.isnan(X).any(axis=1)
    out = sess.run(None, {sess.get_inputs()[0].name: X[ok].astype(np.float32)})
    s[ok] = -out[1].ravel()  # skl2onnx IsolationForest outputs (label, score)
    return s


def smoke_sample(test: list[A.Part]) -> dict:
    """25 normal rows and up to 5 spike rows per series, for the OTA smoke test."""
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for p in test:
        ok = ~np.isnan(p.feats).any(axis=1)
        normal = np.where(ok & (p.y == 0))[0]
        spikes = [f.start for f in p.faults if f.kind == "spike" and ok[f.start]]
        for i in rng.choice(normal, size=min(25, len(normal)), replace=False):
            rows.append(p.feats[i].tolist())
            labels.append(0)
        for i in spikes[:5]:
            rows.append(p.feats[i].tolist())
            labels.append(1)
    return {
        "note": "fixed OTA smoke-test sample: test-year features, spikes labelled 1",
        "features": rows,
        "labels": labels,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    t = time.time()
    train, valid, test = A.build("train"), A.build("valid", seed=1), A.build("test", seed=2)
    print(
        f"data built in {time.time() - t:.0f}s: train {sum(len(p.values) for p in train)} rows, "
        f"valid faults {sum(len(p.faults) for p in valid)}, test faults {sum(len(p.faults) for p in test)}"
    )
    results = {"rules": A.score_events(test, [A.alarms_from_rules(p) for p in test])}

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    onnx_path = OUT / "iforest.onnx"
    A.export_onnx(A.fit_iforest(train), onnx_path)
    # Pick the threshold on the EXPORTED model's scores — the thing that ships.
    # (skl2onnx's score is offset from scikit-learn's score_samples; a threshold
    # from one scale is wrong on the other — found 2026-10-10.)
    threshold = A.pick_threshold(valid, [onnx_scores(onnx_path, p.feats) for p in valid])
    test_scores = [onnx_scores(onnx_path, p.feats) for p in test]
    alarms = [A.alarms_from_scores(p, s, threshold) for p, s in zip(test, test_scores, strict=True)]
    results["iforest"] = {**A.score_events(test, alarms), "threshold": threshold, "bytes": onnx_path.stat().st_size}

    meta = {
        "kind": "iforest",
        "file": onnx_path.name,
        "features": FEATURES,
        "window": WINDOW,
        "threshold": threshold,
        "sha256": A.sha256(onnx_path),
        "version": "local",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    Path("src/aqedge/edge/ota_sample.json").write_text(json.dumps(smoke_sample(test)))
    (RESULTS / "anomaly.json").write_text(json.dumps({"results": results}, indent=2))
    for name, r in results.items():
        print(f"{name:8s}", " ".join(f"{k}={r[k]}" for k in SHOW))


if __name__ == "__main__":
    main()
