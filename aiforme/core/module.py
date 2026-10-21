"""The module interface and the loop that runs modules against the bus.

A module declares a name and the topics it listens to, and turns one event into zero or more new events.
It knows nothing about SQLite or other modules, so a test can call `handle` directly with a hand-made event.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import ClassVar

from aiforme.core.bus import EventBus
from aiforme.core.events import Event

log = logging.getLogger(__name__)

ERROR_TOPIC = "system.error"


class Module(ABC):
    name: ClassVar[str]                       # also the bus consumer name, so it must be unique
    subscribes: ClassVar[tuple[str, ...]]     # topics or patterns, see events.py
    from_start: ClassVar[bool] = False        # replay the whole log on first start

    @abstractmethod
    def handle(self, event: Event) -> Iterable[Event] | None:
        """Reacts to one event; returns (or yields) the events to publish, or None."""


def run_once(bus: EventBus, module: Module, limit: int = 100) -> int:
    """Handles the module's pending events; returns how many it handled.

    An exception in `handle` does not stop the module or block its queue: it becomes a `system.error`
    event and the module moves on, so one bad event cannot jam everything behind it.
    """
    events = bus.poll(module.name, limit)
    for event in events:
        try:
            outputs = list(module.handle(event) or ())
        except Exception as exc:
            log.exception("%s failed on event %s (%s)", module.name, event.id, event.topic)
            outputs = [Event(ERROR_TOPIC, {"module": module.name, "event_id": event.id,
                                           "topic": event.topic, "error": repr(exc)})]
        outputs = [replace(out, source=out.source or module.name) for out in outputs]
        bus.complete(module.name, event, outputs)
    return len(events)


def run(bus: EventBus, modules: Sequence[Module], interval: float = 0.2,
        stop: threading.Event | None = None) -> None:
    """Runs the modules in turn until `stop` is set, sleeping `interval` seconds when all are idle."""
    names = [m.name for m in modules]
    if len(set(names)) != len(names):
        raise ValueError(f"module names must be unique: {names}")
    for module in modules:
        bus.subscribe(module.name, module.subscribes, from_start=module.from_start)
    stop = stop or threading.Event()
    while not stop.is_set():
        handled = sum(run_once(bus, module) for module in modules)
        if not handled:
            stop.wait(interval)
