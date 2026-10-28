"""Labels on heard commands and the model versions trained on them.

A label is my verdict on one `intent.recognized` / `intent.unknown` event: confirmed (the assistant was right)
or corrected (to the intent and slot it should have been). It is pending until a retraining that passes the gate
includes it; from then on it belongs to that version and cannot be changed, so a version always means the same
training data. Version 0 is the classifier trained on the hand-written commands alone.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from aiforme.core.db import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    event_id        INTEGER PRIMARY KEY,     -- the bus event the assistant published for the command
    text            TEXT NOT NULL,
    predicted       TEXT NOT NULL,
    predicted_slots TEXT NOT NULL,
    confidence      REAL NOT NULL,
    served_by       INTEGER NOT NULL,        -- the model version that made the prediction
    heard_at        REAL NOT NULL,
    intent          TEXT NOT NULL,
    slots           TEXT NOT NULL,
    labelled_at     REAL NOT NULL,
    version         INTEGER                  -- the accepted version that first trained on it; NULL: pending
);
CREATE TABLE IF NOT EXISTS versions (
    id          INTEGER PRIMARY KEY,
    created_at  REAL NOT NULL,
    embedder    TEXT NOT NULL,
    accepted    INTEGER NOT NULL,
    new_labels  INTEGER NOT NULL,
    examples    INTEGER NOT NULL,
    metrics     TEXT NOT NULL
);
"""


class LabelLocked(Exception):
    """The label is already part of a model version."""


@dataclass(frozen=True)
class Label:
    event_id: int
    text: str
    predicted: str
    predicted_slots: dict[str, str]
    confidence: float
    served_by: int
    heard_at: float
    intent: str
    slots: dict[str, str]
    labelled_at: float
    version: int | None

    @property
    def verdict(self) -> str:
        return "confirmed" if (self.intent, self.slots) == (self.predicted, self.predicted_slots) else "corrected"

    def to_json(self) -> dict:
        return {"intent": self.intent, "slots": self.slots, "verdict": self.verdict, "version": self.version,
                "labelled_at": self.labelled_at}


@dataclass(frozen=True)
class Version:
    id: int
    created_at: float
    embedder: str
    accepted: bool
    new_labels: int
    examples: int
    metrics: dict

    def to_json(self) -> dict:
        return {"id": self.id, "created_at": self.created_at, "embedder": self.embedder, "accepted": self.accepted,
                "new_labels": self.new_labels, "examples": self.examples, "metrics": self.metrics}


_LABEL_COLUMNS = ("event_id, text, predicted, predicted_slots, confidence, served_by, heard_at, intent, slots, "
                  "labelled_at, version")


class Feedback:
    def __init__(self, path: str | Path = ":memory:", timeout: float = 10.0):
        self._conn = connect(path, SCHEMA, timeout)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Feedback:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- labels -------------------------------------------------------------------------------------------

    def label(self, event_id: int, text: str, predicted: str, predicted_slots: dict, confidence: float,
              served_by: int, heard_at: float, intent: str, slots: dict) -> Label:
        """Adds a label, or replaces a pending one."""
        self._unlocked(event_id)
        self._conn.execute(
            f"INSERT OR REPLACE INTO labels ({_LABEL_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (event_id, text, predicted, json.dumps(predicted_slots, ensure_ascii=False), confidence, served_by,
             heard_at, intent, json.dumps(slots, ensure_ascii=False), time.time()))
        return self.get([event_id])[event_id]

    def remove(self, event_id: int) -> bool:
        self._unlocked(event_id)
        return self._conn.execute("DELETE FROM labels WHERE event_id = ?", (event_id,)).rowcount > 0

    def get(self, event_ids: list[int]) -> dict[int, Label]:
        if not event_ids:
            return {}
        marks = ", ".join("?" for _ in event_ids)
        rows = self._conn.execute(f"SELECT {_LABEL_COLUMNS} FROM labels WHERE event_id IN ({marks})",
                                  list(event_ids)).fetchall()
        return {r[0]: _label(r) for r in rows}

    def labels(self, pending: bool | None = None) -> list[Label]:
        """All labels in the order they were given; `pending` True: not yet in a version, False: in one."""
        where = {None: "", True: "WHERE version IS NULL", False: "WHERE version IS NOT NULL"}[pending]
        rows = self._conn.execute(f"SELECT {_LABEL_COLUMNS} FROM labels {where} ORDER BY labelled_at, event_id")
        return [_label(r) for r in rows.fetchall()]

    def _unlocked(self, event_id: int) -> None:
        row = self._conn.execute("SELECT version FROM labels WHERE event_id = ?", (event_id,)).fetchone()
        if row and row[0] is not None:
            raise LabelLocked(f"the label of event {event_id} is part of version {row[0]}")

    # --- versions -----------------------------------------------------------------------------------------

    def current_version(self) -> int:
        row = self._conn.execute("SELECT MAX(id) FROM versions WHERE accepted = 1").fetchone()
        return row[0] or 0

    def add_version(self, embedder: str, accepted: bool, labels: list[Label], examples: int, metrics: dict,
                    version_id: int | None = None) -> Version:
        """Records a retraining; when accepted, `labels` become part of it. `version_id` is for version 0."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self._conn.execute(
                "INSERT INTO versions (id, created_at, embedder, accepted, new_labels, examples, metrics) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (version_id, time.time(), embedder, int(accepted), len(labels), examples, json.dumps(metrics)))
            if accepted:
                self._conn.executemany("UPDATE labels SET version = ? WHERE event_id = ? AND version IS NULL",
                                       [(cur.lastrowid, lab.event_id) for lab in labels])
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        return next(v for v in self.versions() if v.id == cur.lastrowid)

    def versions(self) -> list[Version]:
        rows = self._conn.execute("SELECT id, created_at, embedder, accepted, new_labels, examples, metrics "
                                  "FROM versions ORDER BY id").fetchall()
        return [Version(i, t, e, bool(a), n, ex, json.loads(m)) for i, t, e, a, n, ex, m in rows]

    def live_accuracy(self) -> list[dict]:
        """Per model version: how many of the commands it served were labelled, and how many it got right."""
        rows = self._conn.execute(
            "SELECT served_by, COUNT(*), SUM(intent = predicted) FROM labels GROUP BY served_by ORDER BY served_by")
        return [{"model": m, "labelled": n, "correct": c} for m, n, c in rows.fetchall()]


def _label(row: tuple) -> Label:
    eid, text, pred, pred_slots, conf, served, heard, intent, slots, at, version = row
    return Label(eid, text, pred, json.loads(pred_slots), conf, served, heard, intent, json.loads(slots), at, version)
