"""Where local state lives. Everything goes under one data directory, which git ignores."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def data_dir() -> Path:
    """`AIFORME_DATA_DIR` if set (`~` is expanded), else `data/` in the repository."""
    custom = os.environ.get("AIFORME_DATA_DIR")
    return Path(custom).expanduser() if custom else ROOT / "data"


def bus_path() -> Path:
    return data_dir() / "bus.sqlite3"


def memory_path() -> Path:
    return data_dir() / "memory.sqlite3"


def calendar_path() -> Path:
    return data_dir() / "calendar.sqlite3"


def modes_path() -> Path:
    return data_dir() / "modes.json"


def activity_path() -> Path:
    return data_dir() / "activity.sqlite3"
