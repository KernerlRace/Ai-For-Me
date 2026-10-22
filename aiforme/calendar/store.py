"""The assistant's own calendar: events in SQLite, no Google account.

An event has a kind, because other modules care about different ones: a `meeting` puts the machine into meeting
mode (mode autopilot, Phase 2), a `shift` is paid work for the finance tracker (Phase 5), `other` is everything
else. Times are local and naive — one person, one machine, one time zone.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

KINDS = ("meeting", "shift", "other")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    start TEXT NOT NULL,
    end   TEXT NOT NULL,
    kind  TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS events_start ON events (start);
"""


@dataclass(frozen=True)
class CalendarEvent:
    id: int
    title: str
    start: datetime
    end: datetime
    kind: str = "other"
    notes: str = ""

    def to_json(self) -> dict:
        return {**asdict(self), "start": self.start.isoformat(timespec="minutes"),
                "end": self.end.isoformat(timespec="minutes")}


def _check(title: str, start: datetime, end: datetime, kind: str) -> None:
    if not title.strip():
        raise ValueError("an event needs a title")
    if end <= start:
        raise ValueError(f"an event must end after it starts: {start} – {end}")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}: expected one of {KINDS}")


class Calendar:
    def __init__(self, path: str | Path = ":memory:", timeout: float = 10.0):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # used from the web server's threads as well as the assistant's; sqlite serialises the writes
        self._conn = sqlite3.connect(path, timeout=timeout, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Calendar:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(self, title: str, start: datetime, end: datetime, kind: str = "other", notes: str = "") -> CalendarEvent:
        _check(title, start, end, kind)
        cur = self._conn.execute(
            "INSERT INTO events (title, start, end, kind, notes) VALUES (?, ?, ?, ?, ?)",
            (title.strip(), _iso(start), _iso(end), kind, notes),
        )
        return self.get(cur.lastrowid)

    def get(self, event_id: int) -> CalendarEvent:
        row = self._conn.execute("SELECT id, title, start, end, kind, notes FROM events WHERE id = ?",
                                 (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"no event {event_id}")
        return _row(row)

    def update(self, event_id: int, **fields) -> CalendarEvent:
        current = self.get(event_id)
        unknown = set(fields) - {"title", "start", "end", "kind", "notes"}
        if unknown:
            raise ValueError(f"cannot change {sorted(unknown)}")
        merged = {**asdict(current), **fields}
        _check(merged["title"], merged["start"], merged["end"], merged["kind"])
        self._conn.execute(
            "UPDATE events SET title = ?, start = ?, end = ?, kind = ?, notes = ? WHERE id = ?",
            (merged["title"].strip(), _iso(merged["start"]), _iso(merged["end"]), merged["kind"], merged["notes"],
             event_id),
        )
        return self.get(event_id)

    def delete(self, event_id: int) -> bool:
        return self._conn.execute("DELETE FROM events WHERE id = ?", (event_id,)).rowcount > 0

    def between(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """Events that overlap [start, end), in start order."""
        rows = self._conn.execute(
            "SELECT id, title, start, end, kind, notes FROM events WHERE start < ? AND end > ? ORDER BY start, id",
            (_iso(end), _iso(start)),
        ).fetchall()
        return [_row(r) for r in rows]


def _iso(t: datetime) -> str:
    # seconds kept, so string order is time order; stored without a time zone
    return t.replace(tzinfo=None, microsecond=0).isoformat(timespec="seconds")


def _row(row: tuple) -> CalendarEvent:
    id_, title, start, end, kind, notes = row
    return CalendarEvent(id_, title, datetime.fromisoformat(start), datetime.fromisoformat(end), kind, notes)
