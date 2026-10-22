"""Where local state lives. Everything goes under one data directory, which git ignores."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def data_dir() -> Path:
    """`AIFORME_DATA_DIR` if set, else `data/` in the repository."""
    return Path(os.environ.get("AIFORME_DATA_DIR") or ROOT / "data")


def bus_path() -> Path:
    return data_dir() / "bus.sqlite3"


def memory_path() -> Path:
    return data_dir() / "memory.sqlite3"


def calendar_path() -> Path:
    return data_dir() / "calendar.sqlite3"
