"""Event bus on SQLite with polling (ROADMAP, Phase 0: "just SQLite with polling to start").

Every event is a row in one table; each consumer keeps a cursor, the id of the last event it finished.
Several processes can share the bus through the same file (WAL mode lets readers and one writer work at
once). Delivery is at least once and in order per consumer: an event is delivered again until the consumer
completes it, and completing it publishes the consumer's outputs and moves its cursor in one transaction,
so a crash never loses an event or publishes its outputs twice.

A new consumer starts at the end of the log: a module that was off does not replay old voice commands when
it starts. `from_start=True` replays everything, for modules that rebuild state from history.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from aiforme.core.events import Event, check_pattern

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT NOT NULL,
    payload    TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cursors (
    consumer TEXT PRIMARY KEY,
    patterns TEXT NOT NULL,
    last_id  INTEGER NOT NULL
);
"""


class EventBus:
    def __init__(self, path: str | Path = ":memory:", timeout: float = 10.0):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: autocommit, transactions are opened explicitly with BEGIN IMMEDIATE
        self._conn = sqlite3.connect(path, timeout=timeout, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EventBus:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def publish(self, event: Event) -> Event:
        with self._transaction():
            return self._insert(event)

    def subscribe(self, consumer: str, patterns: Iterable[str], from_start: bool = False) -> None:
        """Registers a consumer, or updates its patterns; an existing cursor is kept."""
        patterns = [check_pattern(p) for p in patterns]
        if not patterns:
            raise ValueError(f"consumer {consumer!r} subscribes to nothing")
        with self._transaction():
            start = 0 if from_start else self._last_id()
            self._conn.execute(
                "INSERT INTO cursors (consumer, patterns, last_id) VALUES (?, ?, ?) "
                "ON CONFLICT(consumer) DO UPDATE SET patterns = excluded.patterns",
                (consumer, json.dumps(patterns), start),
            )

    def poll(self, consumer: str, limit: int = 100) -> list[Event]:
        """The consumer's next events, oldest first. They stay pending until `complete`."""
        row = self._conn.execute(
            "SELECT patterns, last_id FROM cursors WHERE consumer = ?", (consumer,)
        ).fetchone()
        if row is None:
            raise KeyError(f"consumer {consumer!r} is not subscribed")
        patterns, last_id = json.loads(row[0]), row[1]
        # patterns are checked to [a-z0-9_.*], so they are safe as GLOB patterns as they are
        where = " OR ".join("topic GLOB ?" for _ in patterns)
        rows = self._conn.execute(
            f"SELECT id, topic, payload, source, created_at FROM events "
            f"WHERE id > ? AND ({where}) ORDER BY id LIMIT ?",
            (last_id, *patterns, limit),
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def complete(self, consumer: str, event: Event, outputs: Iterable[Event] = ()) -> list[Event]:
        """Publishes the outputs of handling `event` and moves the consumer past it, atomically."""
        if event.id is None:
            raise ValueError("cannot complete an event that was never published")
        with self._transaction():
            published = [self._insert(out) for out in outputs]
            cur = self._conn.execute(
                "UPDATE cursors SET last_id = MAX(last_id, ?) WHERE consumer = ?", (event.id, consumer)
            )
            if cur.rowcount == 0:
                raise KeyError(f"consumer {consumer!r} is not subscribed")
        return published

    def recent(self, limit: int = 20, pattern: str = "*") -> list[Event]:
        """The last events matching `pattern`, oldest first; for inspection, it moves no cursor."""
        rows = self._conn.execute(
            "SELECT id, topic, payload, source, created_at FROM events "
            "WHERE topic GLOB ? ORDER BY id DESC LIMIT ?",
            (check_pattern(pattern), limit),
        ).fetchall()
        return [_row_to_event(r) for r in reversed(rows)]

    def _insert(self, event: Event) -> Event:
        cur = self._conn.execute(
            "INSERT INTO events (topic, payload, source, created_at) VALUES (?, ?, ?, ?)",
            (event.topic, json.dumps(event.payload, ensure_ascii=False), event.source, event.created_at),
        )
        return replace(event, id=cur.lastrowid)

    def _last_id(self) -> int:
        return self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]

    def _transaction(self) -> _Transaction:
        return _Transaction(self._conn)


class _Transaction:
    """BEGIN IMMEDIATE ... COMMIT, ROLLBACK on error. IMMEDIATE takes the write lock up front, so two
    processes never both read a cursor and then fail to upgrade to a write."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type, exc, tb) -> None:
        self._conn.execute("ROLLBACK" if exc_type else "COMMIT")


def _row_to_event(row: tuple) -> Event:
    id_, topic, payload, source, created_at = row
    return Event(topic=topic, payload=json.loads(payload), source=source, id=id_, created_at=created_at)
