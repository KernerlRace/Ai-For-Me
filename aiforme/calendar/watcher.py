"""Turns calendar time into bus events, so other modules react to the calendar without polling it.

    calendar.event.soon      `lead` before an event starts (meeting mode gets ready, a reminder is said)
    calendar.event.started   when it starts
    calendar.event.ended     when it ends

Each fires once per event. An event edited after it was announced is announced again for its new times; one
already under way when the watcher starts gets `started` at once, but not `soon`.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timedelta

from aiforme.calendar.store import Calendar, CalendarEvent
from aiforme.core import Event, EventBus

SOURCE = "calendar"


class CalendarWatcher:
    def __init__(self, calendar: Calendar, lead: timedelta = timedelta(minutes=2),
                 now: Callable[[], datetime] = datetime.now):
        self.calendar, self.lead, self.now = calendar, lead, now
        self._sent: set[tuple] = set()      # (event id, start, end, what) already published
        self._started_at = now()

    def check(self) -> list[Event]:
        """The announcements due now, each once."""
        t = self.now()
        out = []
        for ev in self.calendar.between(t - timedelta(days=1), t + self.lead + timedelta(seconds=1)):
            due = [("soon", ev.start - self.lead), ("started", ev.start), ("ended", ev.end)]
            for what, when in due:
                key = (ev.id, ev.start, ev.end, what)
                if when > t or key in self._sent:
                    continue
                self._sent.add(key)
                if what == "soon" and ev.start <= self._started_at:
                    continue                    # it was already on when we started
                if what in ("started", "ended") and ev.end <= self._started_at:
                    continue                    # it was over before we started
                out.append(_announce(ev, what, t))
        return out

    def run(self, bus: EventBus, stop: threading.Event, interval: float = 15.0) -> None:
        while not stop.is_set():
            for event in self.check():
                bus.publish(event)
            stop.wait(interval)


def _announce(ev: CalendarEvent, what: str, now: datetime) -> Event:
    payload = {**ev.to_json(), "minutes_to_start": max(0, round((ev.start - now).total_seconds() / 60))}
    return Event(f"calendar.event.{what}", payload, source=SOURCE)
