"""The activity log: one row per sample, with the mode in force — the data a learned mode model trains on.

Window titles are kept (they say which project or video it is) but never leave the machine: the log lives in
the git-ignored data folder like everything else.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from aiforme.mode_autopilot.activity import Activity

SCHEMA = """
CREATE TABLE IF NOT EXISTS activity (
    time       REAL NOT NULL,
    foreground TEXT NOT NULL,
    title      TEXT NOT NULL,
    category   TEXT NOT NULL,
    running    TEXT NOT NULL,
    games      TEXT NOT NULL,
    mode       TEXT NOT NULL,
    manual     INTEGER NOT NULL     -- 1 when the mode was asked for by voice: a correction of the rules
);
CREATE INDEX IF NOT EXISTS activity_time ON activity (time);
"""


class ActivityLog:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=10, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def add(self, a: Activity, mode: str, manual: bool = False) -> None:
        self._conn.execute(
            "INSERT INTO activity VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (a.time, a.foreground, a.title, a.category, json.dumps(sorted(a.running)), json.dumps(list(a.games)),
             mode, int(manual)),
        )

    def rows(self, since: float = 0.0) -> list[tuple[Activity, str, bool]]:
        rows = self._conn.execute(
            "SELECT time, foreground, title, category, running, games, mode, manual FROM activity "
            "WHERE time >= ? ORDER BY time", (since,)).fetchall()
        return [(Activity(t, fg, title, cat, frozenset(json.loads(run)), tuple(json.loads(games))), mode, bool(man))
                for t, fg, title, cat, run, games, mode, man in rows]

    def close(self) -> None:
        self._conn.close()
