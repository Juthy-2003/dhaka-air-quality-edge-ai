"""Train, evaluate and export the edge anomaly models.

Split by time, never randomly:
  train  2022-08 → 2023-12-31  clean series (all four locations report; the
         models are unsupervised)
  valid  2024          faults injected (seed 1) — used ONLY to pick thresholds
  test   2025-01..11   faults injected (seed 2) — reported once

Sensors: PM2.5 and PM10 at the four Dhaka-area locations (8 series); each
sensor's neighbours are the other three locations measuring the same pollutant.
Detectors:
  rules      the baseline (rules.py)
  iforest    scikit-learn IsolationForest on the features, exported to ONNX
Dropout (null readings) is flagged by rule in both.

Event-level scoring: a fault counts as detected if an alarm fires inside it
or within TOLERANCE readings after; an alarm episode (consecutive alarms) is
a true positive if it overlaps a fault (± tolerance).
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from aqedge.data import replay
from aqedge.edge import faults as F
from aqedge.edge.features import FEATURES, WINDOW, peer_log_ratio, series_features
from aqedge.edge.rules import rule_flags

TOLERANCE = 2
SPLITS = {"train": (None, "2023-12-31 23:00"), "valid": ("2024-01-01", "2024-12-31 23:00"), "test": ("2025-01-01", None)}
LOCATIONS = ("Dhaka", "Azimpur", "Paltan", "Tungi")
PARAMS = ("pm25", "pm10")


@dataclass
class Part:
    sensor: str
    values: np.ndarray
    feats: np.ndarray
    faults: list
    y: np.ndarray


TRAIN_FROM = "2022-08-05"  # first full day on which all four locations report


def _matrix(df: pd.DataFrame, param: str, split: str) -> pd.DataFrame:
    """Aligned hourly values, one column per location, for one pollutant."""
    start, end = SPLITS[split]
    cols = {loc: replay.series(df, loc, param) for loc in LOCATIONS}
    m = pd.DataFrame(cols).loc[start or TRAIN_FROM : end]
    return m.dropna(how="all")


def build(split: str, seed: int | None = None, faults_per_sensor: int = 40) -> list[Part]:
    """Faults are injected per sensor BEFORE peer ratios are computed, so a
    faulty sensor also distorts what its neighbours see — as in reality."""
    df = replay.load()
    parts = []
    for p_i, param in enumerate(PARAMS):
        m = _matrix(df, param, split)
        faulty, fault_lists = {}, {}
        for l_i, loc in enumerate(LOCATIONS):
            vals = m[loc].to_numpy(dtype=float)
            flist: list = []
            if seed is not None:
                vals, flist = F.inject(vals, faults_per_sensor, seed=seed * 100 + p_i * 10 + l_i)
            faulty[loc], fault_lists[loc] = vals, flist
        for loc in LOCATIONS:
            others = np.column_stack([faulty[o] for o in LOCATIONS if o != loc])
            ratios = np.array([peer_log_ratio(x, row) for x, row in zip(faulty[loc], others, strict=True)])
            vals = faulty[loc]
            parts.append(Part(f"{loc}/{param}", vals, series_features(vals, ratios), fault_lists[loc],
                              F.labels(len(vals), fault_lists[loc])))
    return parts


def fit_iforest(train: list[Part]) -> IsolationForest:
    X = np.vstack([p.feats for p in train])
    X = X[~np.isnan(X).any(axis=1)]
    return IsolationForest(n_estimators=100, max_samples=4096, random_state=0).fit(X)


def alarms_from_scores(part: Part, scores: np.ndarray, threshold: float) -> np.ndarray:
    a = np.nan_to_num(scores, nan=-np.inf) > threshold
    return a | np.isnan(part.values)  # dropout rule always on


def alarms_from_rules(part: Part) -> np.ndarray:
    return np.array([rule_flags(f, v)[0] for f, v in zip(part.feats, part.values, strict=True)])


def _episodes(a: np.ndarray) -> list[tuple[int, int]]:
    eps, start = [], None
    for i, v in enumerate(a):
        if v and start is None:
            start = i
        elif not v and start is not None:
            eps.append((start, i))
            start = None
    if start is not None:
        eps.append((start, len(a)))
    return eps


def score_events(parts: list[Part], alarms: list[np.ndarray]) -> dict:
    tp_eps = fp_eps = 0
    by_kind = {k: [0, 0] for k in F.KINDS}  # detected, total
    for part, a in zip(parts, alarms, strict=True):
        for f in part.faults:
            hit = a[f.start : min(len(a), f.end + TOLERANCE)].any()
            by_kind[f.kind][0] += int(hit)
            by_kind[f.kind][1] += 1
        for s, e in _episodes(a):
            if any(s < f.end + TOLERANCE and e > f.start - TOLERANCE for f in part.faults):
                tp_eps += 1
            else:
                fp_eps += 1
    detected = sum(v[0] for v in by_kind.values())
    total = sum(v[1] for v in by_kind.values())
    recall = detected / total if total else float("nan")
    precision = tp_eps / (tp_eps + fp_eps) if tp_eps + fp_eps else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarm_episodes": fp_eps,
        **{f"recall_{k}": round(v[0] / v[1], 4) if v[1] else None for k, v in by_kind.items()},
    }


def pick_threshold(parts: list[Part], scores: list[np.ndarray]) -> float:
    """Threshold that maximises event F1 on the validation year."""
    allv = np.concatenate([s[~np.isnan(s)] for s in scores])
    best, best_f1 = None, -1.0
    for q in np.linspace(0.90, 0.9995, 60):
        t = float(np.quantile(allv, q))
        f1 = score_events(parts, [alarms_from_scores(p, s, t) for p, s in zip(parts, scores, strict=True)])["f1"]
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def export_onnx(model: IsolationForest, path: Path) -> None:
    from skl2onnx import to_onnx

    sample = np.zeros((1, len(FEATURES)), dtype=np.float32)
    onx = to_onnx(model, sample, target_opset={"": 17, "ai.onnx.ml": 3})
    path.write_bytes(onx.SerializeToString())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
