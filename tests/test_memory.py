from aiforme.core import Memory


def test_set_get_overwrite_delete(tmp_path):
    with Memory(tmp_path / "memory.sqlite3") as memory:
        assert memory.get("finance", "hourly_rate") is None
        assert memory.get("finance", "hourly_rate", 0) == 0
        memory.set("finance", "hourly_rate", 12.5)
        memory.set("finance", "hourly_rate", 13.0)
        assert memory.get("finance", "hourly_rate") == 13.0
        assert memory.delete("finance", "hourly_rate")
        assert not memory.delete("finance", "hourly_rate")


def test_namespaces_are_separate_and_values_keep_their_shape(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with Memory(path) as memory:
        memory.set("finance", "goal", {"name": "ноутбук", "target": 1200, "tags": ["tech"]})
        memory.set("mode_autopilot", "goal", "coding")
    with Memory(path) as reopened:
        assert reopened.items("finance") == {"goal": {"name": "ноутбук", "target": 1200, "tags": ["tech"]}}
        assert reopened.get("mode_autopilot", "goal") == "coding"
        assert reopened.namespaces() == ["finance", "mode_autopilot"]
