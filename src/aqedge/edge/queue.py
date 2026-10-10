"""Store-and-forward queue on the gateway (SQLite file).

Anomalies are appended first, sent oldest-first when the uplink is up, and
deleted only after the API acknowledges them, so nothing is lost across
restarts or outages. Delivery is at-least-once; the API is idempotent on
(gateway, event_id), so a resend after a lost acknowledgement is harmless.
"""

import json
import sqlite3
import time
from pathlib import Path


class Outbox:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS outbox "
            "(id INTEGER PRIMARY KEY, event_id TEXT UNIQUE, body TEXT, queued_at REAL)"
        )

    def put(self, event: dict) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO outbox (event_id, body, queued_at) VALUES (?, ?, ?)",
            (event["event_id"], json.dumps(event, default=str), time.time()),
        )

    def peek(self, n: int = 50) -> list[tuple[int, dict, float]]:
        rows = self.db.execute("SELECT id, body, queued_at FROM outbox ORDER BY id LIMIT ?", (n,)).fetchall()
        return [(r[0], json.loads(r[1]), r[2]) for r in rows]

    def ack(self, row_id: int) -> None:
        self.db.execute("DELETE FROM outbox WHERE id = ?", (row_id,))

    def depth(self) -> int:
        return self.db.execute("SELECT count(*) FROM outbox").fetchone()[0]
