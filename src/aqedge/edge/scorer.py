"""Load the exported model (ONNX + meta.json) and score one feature vector.

Kept tiny on purpose: this is what runs on the gateway (numpy +
onnxruntime only, no scikit-learn).
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class Scorer:
    def __init__(self, model_dir: Path):
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # one core is the gateway budget
        self.sess = ort.InferenceSession(str(self.dir / self.meta["file"]), opts, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.threshold = float(self.meta["threshold"])

    @property
    def version(self) -> str:
        return str(self.meta.get("version", "local"))

    def score(self, feats: np.ndarray) -> float:
        """Higher = more anomalous. skl2onnx's IsolationForest returns
        (label, score); the score is negated so that alarms are score > threshold."""
        x = feats.reshape(1, -1).astype(np.float32)
        return float(-self.sess.run(None, {self.input: x})[1].ravel()[0])
