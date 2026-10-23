from pathlib import Path

from aiforme.core import config


def test_data_dir_defaults_to_the_repository(monkeypatch):
    monkeypatch.delenv("AIFORME_DATA_DIR", raising=False)
    assert config.data_dir() == config.ROOT / "data"
    assert config.bus_path() == config.ROOT / "data" / "bus.sqlite3"


def test_data_dir_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AIFORME_DATA_DIR", str(tmp_path))
    assert config.calendar_path() == tmp_path / "calendar.sqlite3"
    monkeypatch.setenv("AIFORME_DATA_DIR", "~/aiforme-data")
    assert config.data_dir() == Path.home() / "aiforme-data"
