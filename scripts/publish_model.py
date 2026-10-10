"""Publish an edge model version: copy it next to the manifest and point the
manifest at it. Gateways pick it up on their next OTA poll.

    uv run python scripts/publish_model.py            # next version of models/current
    uv run python scripts/publish_model.py --version 7 --model-dir some/other/model

Result (served by the app at /models/):
    models/published/v3/iforest.onnx
    models/published/manifest.json   {"version": "3", "url": "v3/iforest.onnx", "sha256": "…", "threshold": …}

The manifest is written to a temporary file and renamed, so a gateway never
reads a half-written manifest.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from aqedge import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/current")
    ap.add_argument("--version", help="default: one more than the current manifest")
    a = ap.parse_args()
    out = config.PUBLISHED_MODELS
    manifest_path = out / "manifest.json"
    current = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"version": "0"}
    version = a.version or str(int(current["version"]) + 1)

    meta = json.loads((Path(a.model_dir) / "meta.json").read_text())
    src = Path(a.model_dir) / meta["file"]
    (out / f"v{version}").mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out / f"v{version}" / src.name)
    manifest = {
        "version": version,
        "url": f"v{version}/{src.name}",  # relative to the manifest's own URL
        "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "threshold": meta["threshold"],
    }
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
