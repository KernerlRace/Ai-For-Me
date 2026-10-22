"""The autopilot loop: every few seconds it samples the machine, reads the bus and switches modes.

Reads:     calendar.event.soon / started / ended   (meetings; a spoken reminder before anything starts)
           mode.requested                          (voice: «Амадеус, игровой режим»)
Publishes: mode.changed {mode, previous, reason, actions}
           assistant.reply                         (reminders, and confirming a mode asked for by voice)

Every sample is also written to the activity log with the mode the rules chose: the training data for a
learned mode model (Phase 2, part 3), with voice requests as the corrections.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from aiforme.core import Event, EventBus
from aiforme.mode_autopilot.actions import ModeActions
from aiforme.mode_autopilot.activity import Activity
from aiforme.mode_autopilot.log import ActivityLog
from aiforme.mode_autopilot.rules import ModeDecider

log = logging.getLogger(__name__)

NAME = "mode_autopilot"
CONFIRM = {
    "ru": {"normal": "Включён обычный режим.", "coding": "Включён режим кодинга.",
           "gaming": "Включён игровой режим.", "meeting": "Включён режим встречи."},
    "en": {"normal": "Normal mode on.", "coding": "Coding mode on.", "gaming": "Gaming mode on.",
           "meeting": "Meeting mode on."},
}


class Autopilot:
    def __init__(self, sample: Callable[[], Activity], actions: ModeActions, decider: ModeDecider | None = None,
                 activity_log: ActivityLog | None = None, clock: Callable[[], float] = time.time):
        self.sample, self.actions, self.log, self.clock = sample, actions, activity_log, clock
        self.decider = decider or ModeDecider()
        self.meetings: set[int] = set()
        self._previous = self.decider.mode

    def step(self, bus: EventBus) -> list[Event]:
        """One tick: handle bus events, take a sample, decide; returns what it published."""
        out, requested_lang = [], None
        for event in bus.poll(NAME):
            outputs, lang = self._on_event(event)
            requested_lang = lang or requested_lang
            bus.complete(NAME, event, outputs)
            out += outputs
        now = self.clock()
        try:
            activity = self.sample()
        except Exception:
            log.exception("could not sample the activity")
            activity = None
        decision = self.decider.update(activity, now, meeting_on=bool(self.meetings))
        if activity is not None and self.log is not None:
            self.log.add(activity, self.decider.mode, manual=self.decider.override is not None)
        if decision:
            previous = self._previous
            done = self.actions.switch(decision.mode)
            changed = Event("mode.changed", {"mode": decision.mode, "previous": previous, "reason": decision.reason,
                                             "actions": done}, source=NAME)
            out.append(bus.publish(changed))
            if requested_lang:
                text = CONFIRM.get(requested_lang, CONFIRM["ru"])[decision.mode]
                out.append(bus.publish(Event("assistant.reply", {"text": text}, source=NAME)))
        self._previous = self.decider.mode
        return out

    def _on_event(self, event: Event) -> tuple[list[Event], str | None]:
        p = event.payload
        if event.topic == "mode.requested":
            try:
                self.decider.request(p.get("mode", ""), self.clock())
            except ValueError:
                return [], None
            return [], p.get("language", "ru")
        if event.topic == "calendar.event.soon":
            minutes = p.get("minutes_to_start", 0)
            text = f"Через {minutes} мин.: {p.get('title', '')}."
            return [Event("assistant.reply", {"text": text}, source=NAME)], None
        if p.get("kind") != "meeting":
            return [], None
        if event.topic == "calendar.event.started":
            self.meetings.add(p["id"])
        elif event.topic == "calendar.event.ended":
            self.meetings.discard(p["id"])
        return [], None

    def run(self, bus: EventBus, stop: threading.Event, interval: float = 5.0) -> None:
        bus.subscribe(NAME, ["calendar.event.*", "mode.requested"])
        try:
            while not stop.is_set():
                self.step(bus)
                stop.wait(interval)
        finally:
            done = self.actions.restore()            # leave the machine as it was found
            if done:
                log.info("restored on exit: %s", ", ".join(done))
