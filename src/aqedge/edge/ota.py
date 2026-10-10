"""Over-the-air model updates from a manifest file.

The app serves `models/manifest.json`:

    {"version": "2", "url": "v2/iforest.onnx", "sha256": "…", "threshold": 0.158}

(`scripts/publish_model.py` writes it.) Every POLL_SECONDS the gateway:
  1. downloads the manifest; if its version is the one running, stops here;
  2. downloads the model file (url is relative to the manifest) into a staging dir;
  3. checks the file's sha256 against the manifest;
  4. loads it and scores a fixed sample — finite scores, few alarms on normal
     rows, most spikes flagged (a smoke test, not an evaluation);
  5. swaps directories (rename), keeping the old one as `.previous`;
  6. tells the agent to reload; if that fails, moves `.previous` back.

Plain HTTP with `requests` — no registry client on the gateway.
"""

import hashlib
import json
import logging
import os
import shutil
import threading
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests

from aqedge.edge.scorer import Scorer

log = logging.getLogger("edge.ota")
MANIFEST_URL = os.environ.get("MODEL_MANIFEST_URL", "http://localhost:8000/models/manifest.json")
POLL_SECONDS = float(os.environ.get("OTA_POLL_SECONDS", "30"))
SAMPLE = Path(__file__).with_name("ota_sample.json")


class Updater:
    def __init__(self, model_dir: Path, reload_fn, manifest_url: str = MANIFEST_URL, http=None):
        self.model_dir = Path(model_dir)
        self.reload = reload_fn
        self.manifest_url = manifest_url
        self.http = http or requests.Session()

    def manifest(self) -> dict | None:
        r = self.http.get(self.manifest_url, timeout=10)
        if r.status_code == 404:  # nothing published yet
            return None
        r.raise_for_status()
        return r.json()

    def running_version(self) -> str | None:
        meta = self.model_dir / "meta.json"
        return str(json.loads(meta.read_text()).get("version")) if meta.exists() else None

    def update(self) -> str:
        m = self.manifest()
        if m is None or str(m["version"]) == self.running_version():
            return "unchanged"
        version = str(m["version"])
        staging = self.model_dir.with_name(self.model_dir.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        r = self.http.get(urljoin(self.manifest_url, m["url"]), timeout=60)
        r.raise_for_status()
        if hashlib.sha256(r.content).hexdigest() != m["sha256"]:
            shutil.rmtree(staging)
            log.error("OTA v%s rejected: sha256 mismatch", version)
            return "rejected:checksum"
        (staging / "model.onnx").write_bytes(r.content)
        meta = {"file": "model.onnx", "threshold": m["threshold"], "version": version, "sha256": m["sha256"]}
        (staging / "meta.json").write_text(json.dumps(meta, indent=2))
        if not self.smoke_test(staging):
            shutil.rmtree(staging)
            log.error("OTA v%s rejected: smoke test failed", version)
            return "rejected:smoke"
        previous = self.model_dir.with_name(self.model_dir.name + ".previous")
        shutil.rmtree(previous, ignore_errors=True)
        if self.model_dir.exists():
            os.replace(self.model_dir, previous)
        os.replace(staging, self.model_dir)  # atomic on one filesystem
        try:
            running = self.reload()
            log.warning("OTA: now running model v%s", running)
            return f"updated:{running}"
        except Exception:
            log.exception("OTA: new model failed to load; rolling back")
            os.replace(self.model_dir, staging)
            os.replace(previous, self.model_dir)
            self.reload()
            return "rolled-back"

    @staticmethod
    def smoke_test(model_dir: Path) -> bool:
        """Finite scores on a fixed sample, and alarms on the injected rows."""
        sample = json.loads(SAMPLE.read_text())
        try:
            scorer = Scorer(model_dir)
            scores = np.array([scorer.score(np.array(f, dtype=np.float32)) for f in sample["features"]])
        except Exception:  # e.g. not a valid ONNX file
            log.exception("OTA: candidate model does not load or score")
            return False
        if not np.isfinite(scores).all():
            return False
        flagged = scores > scorer.threshold
        labels = np.array(sample["labels"])
        return flagged[labels == 0].mean() <= 0.2 and flagged[labels == 1].mean() >= 0.5

    def loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                result = self.update()
                if result != "unchanged":
                    log.warning("OTA: %s", result)
            except requests.RequestException as exc:
                log.info("OTA: manifest unreachable (%s)", type(exc).__name__)
            except Exception:  # never let one bad update kill the updater thread
                log.exception("OTA: update failed; keeping the running model")
            stop.wait(POLL_SECONDS)
