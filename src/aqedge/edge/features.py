"""Per-reading features from a short window of one sensor's history and its
neighbours.

Pure numpy, no pandas: the edge agent imports exactly this function, and so
does training, so the model sees identical features in both places.

For a new value x, the sensor's previous values `hist`, and the matching
history of "peer log-ratios" (log of this sensor over the median of the other
sensors measuring the same pollutant at the same time):

  z            how many standard deviations x is from the window mean
  rel_delta    jump from the previous value, relative to the window mean
  log_ratio    log(x / window median)                     — spikes
  flat_len     how many previous values equal x             — stuck sensors
  short_std    std of the last 6 values incl. x, relative   — ~0 when stuck
  peer_dev     today's peer log-ratio minus its window mean — sensor vs neighbours
  peer_dev_6   mean peer_dev over the last 6 readings       — slow drift

A real pollution episode moves every sensor together, so peer_dev stays near
0; a drifting sensor slowly diverges from its neighbours.
"""

import numpy as np

WINDOW = 48
FEATURES = ["z", "rel_delta", "log_ratio", "flat_len", "short_std", "peer_dev", "peer_dev_6"]
EPS = 1e-6


def peer_log_ratio(x: float, peers: np.ndarray) -> float:
    p = peers[~np.isnan(peers)]
    if np.isnan(x) or len(p) == 0:
        return np.nan
    return float(np.log((x + 1) / (np.median(p) + 1)))


def features(hist: np.ndarray, x: float, peer_hist: np.ndarray, peer_now: float) -> np.ndarray:
    h = hist[~np.isnan(hist)][-WINDOW:]
    if len(h) < 8 or np.isnan(x):
        return np.full(len(FEATURES), np.nan, dtype=np.float32)
    mean, std, med = h.mean(), h.std(), np.median(h)
    flat = 0
    for v in h[::-1]:
        if abs(v - x) < EPS:
            flat += 1
        else:
            break
    last6 = np.append(h[-5:], x)
    ph = peer_hist[~np.isnan(peer_hist)][-WINDOW:]
    if len(ph) >= 8 and not np.isnan(peer_now):
        base = ph.mean()
        devs = np.append(peer_hist[-5:], peer_now) - base
        peer_dev, peer_dev_6 = peer_now - base, float(np.nanmean(devs))
    else:
        peer_dev = peer_dev_6 = 0.0  # no neighbours: neutral
    return np.array(
        [
            (x - mean) / (std + EPS),
            (x - h[-1]) / (mean + EPS),
            np.log((x + 1) / (med + 1)),
            min(flat, 24),
            last6.std() / (mean + EPS),
            peer_dev,
            peer_dev_6,
        ],
        dtype=np.float32,
    )


def series_features(values: np.ndarray, peer_ratios: np.ndarray) -> np.ndarray:
    """Features for every position (row i uses positions < i as history)."""
    out = np.full((len(values), len(FEATURES)), np.nan, dtype=np.float32)
    for i in range(len(values)):
        lo = max(0, i - WINDOW)
        out[i] = features(values[lo:i], values[i], peer_ratios[lo:i], peer_ratios[i])
    return out
