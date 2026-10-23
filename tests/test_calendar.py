from datetime import datetime, timedelta

import pytest

from aiforme.calendar import Calendar, CalendarEvent
from aiforme.calendar.watcher import CalendarWatcher

T = datetime(2026, 10, 23, 15, 0)


@pytest.fixture
def cal(tmp_path):
    with Calendar(tmp_path / "calendar.sqlite3") as c:
        yield c


def test_add_get_update_delete(cal):
    ev = cal.add("Standup", T, T + timedelta(minutes=15), kind="meeting")
    assert cal.get(ev.id) == ev
    moved = cal.update(ev.id, start=T + timedelta(hours=1), end=T + timedelta(hours=1, minutes=30))
    assert moved.start == T + timedelta(hours=1) and moved.title == "Standup"
    assert cal.delete(ev.id) and not cal.delete(ev.id)
    with pytest.raises(KeyError):
        cal.get(ev.id)


def test_validation(cal):
    with pytest.raises(ValueError):
        cal.add("Backwards", T, T - timedelta(minutes=1))
    with pytest.raises(ValueError):
        cal.add("  ", T, T + timedelta(hours=1))
    with pytest.raises(ValueError):
        cal.add("Party", T, T + timedelta(hours=1), kind="party")
    ev = cal.add("Ok", T, T + timedelta(hours=1))
    with pytest.raises(ValueError):
        cal.update(ev.id, end=T)
    with pytest.raises(ValueError):
        cal.update(ev.id, id=5)


def test_update_trims_the_title_and_keeps_the_other_fields(cal):
    ev = cal.add("  Shift ", T, T + timedelta(hours=8), kind="shift", notes="bring the badge")
    assert ev.title == "Shift"
    renamed = cal.update(ev.id, title="  Night shift  ")
    assert renamed == CalendarEvent(ev.id, "Night shift", ev.start, ev.end, "shift", "bring the badge")
    with pytest.raises(KeyError):
        cal.update(ev.id + 1, title="Nobody")

def test_between_returns_overlapping_events_in_order(cal):
    late = cal.add("Late", T + timedelta(hours=3), T + timedelta(hours=4))
    early = cal.add("Early", T - timedelta(hours=1), T + timedelta(minutes=30))   # began before the range
    cal.add("Tomorrow", T + timedelta(days=1), T + timedelta(days=1, hours=1))
    assert [e.id for e in cal.between(T, T + timedelta(hours=5))] == [early.id, late.id]
    assert cal.between(T + timedelta(minutes=30), T + timedelta(hours=3)) == []   # touching edges do not overlap


def test_events_survive_a_restart(tmp_path):
    with Calendar(tmp_path / "c.sqlite3") as c:
        ev = c.add("Shift", T, T + timedelta(hours=8), kind="shift", notes="£12.50/h")
    with Calendar(tmp_path / "c.sqlite3") as c:
        assert c.get(ev.id) == ev


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def test_watcher_announces_soon_started_ended_once(cal):
    clock = Clock(T - timedelta(minutes=10))
    w = CalendarWatcher(cal, lead=timedelta(minutes=2), now=clock)
    ev = cal.add("Call", T, T + timedelta(minutes=30), kind="meeting")
    assert w.check() == []
    clock.t = T - timedelta(minutes=2)
    [soon] = w.check()
    assert soon.topic == "calendar.event.soon" and soon.payload["id"] == ev.id
    assert soon.payload["minutes_to_start"] == 2 and soon.payload["kind"] == "meeting"
    assert w.check() == []                                  # once
    clock.t = T
    assert [e.topic for e in w.check()] == ["calendar.event.started"]
    clock.t = T + timedelta(minutes=31)
    assert [e.topic for e in w.check()] == ["calendar.event.ended"]
    assert w.check() == []


def test_watcher_on_start_skips_what_is_already_past(cal):
    cal.add("Over", T - timedelta(hours=2), T - timedelta(hours=1))
    on = cal.add("Ongoing", T - timedelta(minutes=10), T + timedelta(minutes=20))
    w = CalendarWatcher(cal, now=Clock(T))
    out = w.check()
    assert [(e.topic, e.payload["id"]) for e in out] == [("calendar.event.started", on.id)]


def test_watcher_announces_a_moved_event_again(cal):
    clock = Clock(T - timedelta(minutes=5))
    w = CalendarWatcher(cal, lead=timedelta(minutes=2), now=clock)
    ev = cal.add("Call", T, T + timedelta(minutes=30))
    clock.t = T - timedelta(minutes=1)
    assert [e.topic for e in w.check()] == ["calendar.event.soon"]
    cal.update(ev.id, start=T + timedelta(hours=1), end=T + timedelta(hours=2))
    clock.t = T + timedelta(minutes=59)
    assert [e.topic for e in w.check()] == ["calendar.event.soon"]


def test_watcher_forgets_events_a_day_after_they_end(cal):
    clock = Clock(T - timedelta(minutes=5))
    w = CalendarWatcher(cal, lead=timedelta(minutes=2), now=clock)
    cal.add("Call", T, T + timedelta(minutes=30))
    for minutes in (-2, 0, 31):
        clock.t = T + timedelta(minutes=minutes)
        w.check()
    assert len(w._sent) == 3
    clock.t = T + timedelta(days=1, minutes=31)
    assert w.check() == [] and w._sent == set()
