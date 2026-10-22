"""Which mode the machine is in, from the last few minutes of activity — the rule-based baseline.

Priority, highest first:
    a mode asked for by voice (until it expires or «автоматический режим» clears it)
    meeting   a calendar meeting is on, or a meeting app has been in front for a minute
    gaming    a game is running
    coding    an IDE or a terminal was in front for at least half of the last five minutes
    normal    anything else

A new mode must hold for `stable_s` before the switch, so alt-tabbing out of a game or glancing at a browser
does not flip the machine back and forth. Voice and the calendar switch at once: they are deliberate.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from aiforme.mode_autopilot.activity import Activity, share

MODES = ("normal", "coding", "gaming", "meeting")


@dataclass(frozen=True)
class Decision:
    mode: str
    reason: str
    immediate: bool = False


def candidate(history: list[Activity], meeting_on: bool, coding_share: float = 0.5,
              coding_window_s: float = 300.0, meeting_app_s: float = 60.0) -> Decision:
    """The mode the latest activity points to, before any smoothing."""
    if meeting_on:
        return Decision("meeting", "a meeting in the calendar", immediate=True)
    if not history:
        return Decision("normal", "no activity yet")
    now = history[-1].time
    last_minute = [a for a in history if a.time >= now - meeting_app_s]
    if (history[0].time <= now - meeting_app_s and share(last_minute, {"meeting"}) >= 0.8):
        return Decision("meeting", f"{history[-1].foreground} in front for a minute")
    if "game" in history[-1].running:
        return Decision("gaming", "game running: " + ", ".join(history[-1].games or ("unknown",)))
    window = [a for a in history if a.time >= now - coding_window_s]
    if len(window) >= 3 and share(window, {"ide", "terminal"}) >= coding_share:
        return Decision("coding", f"an IDE or a terminal in front {share(window, {'ide', 'terminal'}):.0%} "
                                  f"of the last {coding_window_s / 60:.0f} min")
    return Decision("normal", "nothing in particular")


class ModeDecider:
    def __init__(self, stable_s: float = 20.0, history_s: float = 300.0):
        self.stable_s, self.history_s = stable_s, history_s
        self.history: deque[Activity] = deque()
        self.mode, self.reason = "normal", "start"
        self._pending: tuple[str, float] | None = None       # (mode, since)
        self.override: tuple[str, float] | None = None       # (mode, until)

    def request(self, mode: str, now: float, for_s: float = 2 * 3600) -> None:
        """A mode asked for by voice; 'auto' hands control back to the rules."""
        if mode == "auto":
            self.override = None
        elif mode in MODES:
            self.override = (mode, now + for_s)
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def update(self, activity: Activity | None, now: float, meeting_on: bool = False) -> Decision | None:
        """Feeds one sample; returns the decision when the mode changes, else None."""
        if activity is not None:
            self.history.append(activity)
        while self.history and self.history[0].time < now - self.history_s:
            self.history.popleft()
        if self.override and now >= self.override[1]:
            self.override = None
        if self.override:
            decision = Decision(self.override[0], "asked for by voice", immediate=True)
        else:
            decision = candidate(list(self.history), meeting_on)
        if decision.mode == self.mode:
            self._pending = None
            return None
        if not decision.immediate:
            if self._pending is None or self._pending[0] != decision.mode:
                self._pending = (decision.mode, now)
            if now - self._pending[1] < self.stable_s:
                return None
        self.mode, self.reason, self._pending = decision.mode, decision.reason, None
        return decision
