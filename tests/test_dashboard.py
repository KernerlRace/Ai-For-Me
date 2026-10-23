from fastapi.testclient import TestClient

from aiforme.calendar import Calendar
from aiforme.core import EventBus
from aiforme.dashboard.server import create_app


def make(tmp_path):
    bus_path = tmp_path / "bus.sqlite3"
    with EventBus(bus_path) as bus:
        bus.subscribe("watch", ["calendar.*"])
    cal = Calendar(tmp_path / "calendar.sqlite3")
    return TestClient(create_app(cal, bus_path)), cal, bus_path


def test_create_list_update_delete(tmp_path):
    client, cal, bus_path = make(tmp_path)
    r = client.post("/api/events", json={"title": "Standup", "start": "2026-10-23T15:00", "end": "2026-10-23T15:15",
                                         "kind": "meeting"})
    assert r.status_code == 201
    ev = r.json()
    assert ev["title"] == "Standup" and ev["start"] == "2026-10-23T15:00" and ev["kind"] == "meeting"

    week = client.get("/api/events", params={"start": "2026-10-19T00:00", "end": "2026-10-26T00:00"}).json()
    assert [e["id"] for e in week] == [ev["id"]]
    assert client.get("/api/events", params={"start": "2026-10-24T00:00", "end": "2026-10-25T00:00"}).json() == []

    moved = client.patch(f"/api/events/{ev['id']}", json={"start": "2026-10-23T16:00", "end": "2026-10-23T16:30"})
    assert moved.status_code == 200 and moved.json()["start"] == "2026-10-23T16:00"
    assert moved.json()["title"] == "Standup"

    assert client.delete(f"/api/events/{ev['id']}").status_code == 204
    assert client.delete(f"/api/events/{ev['id']}").status_code == 404

    with EventBus(bus_path) as bus:
        actions = [e.payload["action"] for e in bus.poll("watch")]
    assert actions == ["added", "updated", "deleted"]


def test_mode_is_the_last_change(tmp_path):
    from aiforme.core import Event
    client, _, bus_path = make(tmp_path)
    assert client.get("/api/mode").json() == {"mode": None}
    with EventBus(bus_path) as bus:
        bus.publish(Event("mode.changed", {"mode": "coding", "reason": "IDE"}))
        bus.publish(Event("mode.changed", {"mode": "gaming", "reason": "game running: dota2.exe"}))
    body = client.get("/api/mode").json()
    assert body["mode"] == "gaming" and body["reason"] == "game running: dota2.exe"


def test_bad_input_is_rejected(tmp_path):
    client, *_ = make(tmp_path)
    backwards = {"title": "X", "start": "2026-10-23T15:00", "end": "2026-10-23T14:00"}
    assert client.post("/api/events", json=backwards).status_code == 422
    assert client.post("/api/events", json={**backwards, "end": "2026-10-23T16:00", "kind": "party"}).status_code == 422
    assert client.post("/api/events", json={**backwards, "title": "", "end": "2026-10-23T16:00"}).status_code == 422
    assert client.patch("/api/events/999", json={"title": "Y"}).status_code == 404
    backwards_range = {"start": "2026-10-26T00:00", "end": "2026-10-19T00:00"}
    assert client.get("/api/events", params=backwards_range).status_code == 422


def test_times_with_a_time_zone_become_local(tmp_path):
    client, cal, _ = make(tmp_path)
    r = client.post("/api/events", json={"title": "UTC", "start": "2026-10-23T14:00:00Z",
                                         "end": "2026-10-23T15:00:00Z"})
    ev = cal.get(r.json()["id"])
    assert ev.start.tzinfo is None and (ev.end - ev.start).total_seconds() == 3600
