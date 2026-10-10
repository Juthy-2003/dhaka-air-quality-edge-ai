"""Test database: a throwaway database `aq_test_<random>` created on the
Postgres at TEST_DATABASE_URL (default: the compose one on localhost:5435,
started by `make test`; in CI a pgvector service container) and dropped at
the end. Tests that need it are skipped if no Postgres is reachable."""

import os
import tempfile
import uuid

import pytest

BASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://aq:aq@localhost:5435/aq")
TEST_DB = f"aq_test_{uuid.uuid4().hex[:8]}"
os.environ["DATABASE_URL"] = BASE_URL.rsplit("/", 1)[0] + "/" + TEST_DB  # read by aqedge.config at import
os.environ["REPLAY_ENABLED"] = "false"  # tests call the replay step directly
os.environ["PUBLISHED_MODELS"] = tempfile.mkdtemp(prefix="aq-published-")  # the app serves this at /models


@pytest.fixture(scope="session")
def database():
    import psycopg

    try:
        admin = psycopg.connect(BASE_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable at TEST_DATABASE_URL")
    admin.execute(f"CREATE DATABASE {TEST_DB}")
    from aqedge import db

    db.init()
    yield db
    admin.execute(f"DROP DATABASE {TEST_DB} WITH (FORCE)")
    admin.close()
