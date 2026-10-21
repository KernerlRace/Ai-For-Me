"""Events: the only thing modules exchange.

A module never calls another module. It reads events from the bus and publishes new ones, so each module
can be built, tested and restarted on its own (ROADMAP: "event in → event out").

Topics are dotted lowercase names, `<area>.<what happened>`: `voice.transcribed`, `intent.recognized`,
`system.error`. A subscription is an exact topic, a prefix ending in `.*` (`voice.*` matches
`voice.transcribed` and `voice.wake.detected`, but not `voice` itself), or `*` for everything.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

_NAME = r"[a-z0-9_]+(?:\.[a-z0-9_]+)*"
TOPIC_RE = re.compile(rf"^{_NAME}$")
PATTERN_RE = re.compile(rf"^(?:\*|{_NAME}(?:\.\*)?)$")


def check_topic(topic: str) -> str:
    if not TOPIC_RE.match(topic):
        raise ValueError(f"bad topic {topic!r}: expected dotted lowercase names like 'voice.transcribed'")
    return topic


def check_pattern(pattern: str) -> str:
    if not PATTERN_RE.match(pattern):
        raise ValueError(f"bad pattern {pattern!r}: expected a topic, 'prefix.*' or '*'")
    return pattern


@dataclass(frozen=True)
class Event:
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = ""                # module name; the runner fills it in for module outputs
    id: int | None = None           # set by the bus on publish, increasing
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        check_topic(self.topic)
        if not isinstance(self.payload, dict):
            raise TypeError(f"payload must be a dict, got {type(self.payload).__name__}")
