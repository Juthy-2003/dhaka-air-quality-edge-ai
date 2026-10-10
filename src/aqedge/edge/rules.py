"""The rule baseline every model has to beat, and the always-on dropout rule.

Dropout (a null reading) is decided by this rule in every variant — a model
cannot score a value that does not exist.
"""

import numpy as np

STUCK_LEN = 4
SPIKE_Z = 4.0
SPIKE_LOG_RATIO = 0.9  # ≈ 2.5×
DRIFT_PEER_DEV = 0.25  # sustained 25%+ divergence from the neighbours (log scale ≈ 28%)


def rule_flags(feats: np.ndarray, value: float) -> tuple[bool, str]:
    if np.isnan(value):
        return True, "dropout"
    if np.isnan(feats).any():
        return False, ""
    z, _, log_ratio, flat, _, _, peer_dev_6 = feats
    if flat >= STUCK_LEN:
        return True, "stuck"
    if abs(z) > SPIKE_Z or log_ratio > SPIKE_LOG_RATIO:
        return True, "spike"
    if abs(peer_dev_6) > DRIFT_PEER_DEV:
        return True, "drift"
    return False, ""
