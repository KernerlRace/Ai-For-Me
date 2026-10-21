"""Basic long-term memory: JSON values under (namespace, key), in SQLite.

Each module writes under its own namespace (`finance`, `mode_autopilot`, ...) and may read the others.
This is the Phase 0 stand-in for the memory graph of Phase 8, which will add entities and relations;
modules talk to memory only through get/set, so the storage behind it can change.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
"""


class Memory:
    def __init__(self, path: str | Path = ":memory:", timeout: float = 10.0):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=timeout, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def set(self, namespace: str, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO memory (namespace, key, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(namespace, key) DO UPDATE "
            "SET value = excluded.value, updated_at = excluded.updated_at",
            (namespace, key, json.dumps(value, ensure_ascii=False), time.time()),
        )

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM memory WHERE namespace = ? AND key = ?", (namespace, key)
        ).fetchone()
        return default if row is None else json.loads(row[0])

    def delete(self, namespace: str, key: str) -> bool:
        cur = self._conn.execute("DELETE FROM memory WHERE namespace = ? AND key = ?", (namespace, key))
        return cur.rowcount > 0

    def items(self, namespace: str) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT key, value FROM memory WHERE namespace = ? ORDER BY key", (namespace,)
        ).fetchall()
        return {key: json.loads(value) for key, value in rows}

    def namespaces(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT DISTINCT namespace FROM memory ORDER BY namespace")]
