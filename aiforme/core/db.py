"""Opening the SQLite files that hold local state: the bus, memory, the calendar, the activity log."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: str | Path, schema: str, timeout: float = 10.0, **kwargs) -> sqlite3.Connection:
    """An autocommit connection in WAL mode with `schema` applied; the parent folder is created if needed."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None: autocommit, transactions are opened explicitly with BEGIN IMMEDIATE where needed
    conn = sqlite3.connect(path, timeout=timeout, isolation_level=None, **kwargs)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(schema)
    return conn
