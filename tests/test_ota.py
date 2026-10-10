"""OTA against the real manifest endpoint: the app serves models/manifest.json
from PUBLISHED_MODELS (a temp dir, see conftest.py), the gateway's Updater
fetches it through FastAPI's TestClient. No database needed."""

import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from aqedge import config
from aqedge.edge import ota

MANIFEST_URL = "http://testserver/models/manifest.json"


@pytest.fixture
def http():
    from aqedge.app import app

    for p in config.PUBLISHED_MODELS.iterdir():  # start every test with nothing published
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    return TestClient(app)  # no `with`: the lifespan (database, replay thread) is not started


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    shutil.copytree("models/current", d)
    return d


def publish(version: str, model: bytes, sha256: str | None = None) -> None:
    """What scripts/publish_model.py does, with knobs to tamper."""
    (config.PUBLISHED_MODELS / f"v{version}").mkdir()
    (config.PUBLISHED_MODELS / f"v{version}" / "iforest.onnx").write_bytes(model)
    threshold = json.loads(open("models/current/meta.json").read())["threshold"]
    manifest = {
        "version": version,
        "url": f"v{version}/iforest.onnx",
        "threshold": threshold,
        "sha256": sha256 or hashlib.sha256(model).hexdigest(),
    }
    (config.PUBLISHED_MODELS / "manifest.json").write_text(json.dumps(manifest))


def running(model_dir) -> str:
    return json.loads((model_dir / "meta.json").read_text())["version"]


def test_nothing_published_means_unchanged(http, model_dir):
    assert ota.Updater(model_dir, lambda: "x", MANIFEST_URL, http).update() == "unchanged"


def test_publish_script_then_swap_keeps_previous(http, model_dir):
    env = {**os.environ, "PUBLISHED_MODELS": str(config.PUBLISHED_MODELS)}
    subprocess.run([sys.executable, "scripts/publish_model.py", "--version", "7"], check=True, env=env)
    reloads = []
    u = ota.Updater(model_dir, lambda: reloads.append(1) or "7", MANIFEST_URL, http)
    assert u.update() == "updated:7"
    assert running(model_dir) == "7" and reloads == [1]
    assert model_dir.with_name("model.previous").exists()
    assert u.update() == "unchanged"


def test_checksum_mismatch_is_rejected(http, model_dir):
    good = open("models/current/iforest.onnx", "rb").read()
    publish("8", good + b"x", sha256=hashlib.sha256(good).hexdigest())  # corrupted in transit
    assert ota.Updater(model_dir, lambda: "8", MANIFEST_URL, http).update() == "rejected:checksum"
    assert running(model_dir) == "local"


def test_broken_model_never_replaces_the_running_one(http, model_dir):
    publish("9", b"not an onnx model")  # checksum matches, but it is not a model
    assert ota.Updater(model_dir, lambda: "9", MANIFEST_URL, http).update() == "rejected:smoke"
    assert running(model_dir) == "local"


def test_reload_failure_rolls_back(http, model_dir):
    publish("10", open("models/current/iforest.onnx", "rb").read())
    calls = []

    def reload():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("agent could not load it")
        return "local"

    assert ota.Updater(model_dir, reload, MANIFEST_URL, http).update() == "rolled-back"
    assert running(model_dir) == "local"
