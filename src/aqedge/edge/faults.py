"""Inject SYNTHETIC sensor faults into a clean series, with ground-truth labels.

The replay data is modelled and smooth, so every fault here is known exactly
— which is what lets us measure precision and recall per fault type. Real
low-cost sensors fail in these four recognisable ways:

  stuck    the value freezes for 6–24 readings     (firmware hang)
  spike    1–2 readings jump to 3–6× the level     (electrical noise)
  drift    an offset grows to +40–80% over 48–96   (ageing, dirty optics)
  dropout  readings arrive as null for 3–12        (power or link loss)
"""

from dataclasses import dataclass

import numpy as np

KINDS = ("stuck", "spike", "drift", "dropout")


@dataclass
class Fault:
    kind: str
    start: int
    end: int  # exclusive


def inject(values: np.ndarray, n_faults: int, seed: int, min_gap: int = 120) -> tuple[np.ndarray, list[Fault]]:
    rng = np.random.default_rng(seed)
    out = values.astype(float).copy()
    faults: list[Fault] = []
    pos = 200  # leave room for the feature window
    while len(faults) < n_faults:
        pos += int(rng.integers(min_gap, min_gap * 3))
        if pos > len(out) - 120:  # keep every fault (longest: 96 readings) inside the series
            break
        kind = KINDS[len(faults) % len(KINDS)]
        if kind == "stuck":
            n = int(rng.integers(6, 25))
            out[pos : pos + n] = out[pos - 1]
        elif kind == "spike":
            n = int(rng.integers(1, 3))
            out[pos : pos + n] = out[pos : pos + n] * rng.uniform(3, 6)
        elif kind == "drift":
            n = int(rng.integers(48, 97))
            out[pos : pos + n] = out[pos : pos + n] * (1 + np.linspace(0, rng.uniform(0.4, 0.8), n))
        else:
            n = int(rng.integers(3, 13))
            out[pos : pos + n] = np.nan
        faults.append(Fault(kind, pos, pos + n))
        pos += n
    return out, faults


def labels(length: int, faults: list[Fault]) -> np.ndarray:
    y = np.zeros(length, dtype=np.int8)
    for f in faults:
        y[f.start : f.end] = 1
    return y
