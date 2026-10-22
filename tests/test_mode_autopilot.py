import pytest

from aiforme.core import Event, EventBus
from aiforme.mode_autopilot.actions import DEFAULTS, HIGH_PERFORMANCE, ModeActions, load_settings
from aiforme.mode_autopilot.activity import Activity, categorize
from aiforme.mode_autopilot.autopilot import NAME, Autopilot
from aiforme.mode_autopilot.log import ActivityLog
from aiforme.mode_autopilot.rules import ModeDecider


def test_categorize():
    assert categorize("Code.exe") == "ide"
    steam_game = r"C:\Program Files (x86)\Steam\steamapps\common\World of Tanks Blitz\wotblitz.exe"
    assert categorize("wotblitz.exe", steam_game) == "game"
    assert categorize("anything.exe", r"D:\Games\Epic Games\Fortnite\x.exe") == "game"
    assert categorize("steam.exe", r"C:\Program Files (x86)\Steam\steam.exe") == "other"      # a launcher
    assert categorize("steamwebhelper.exe", r"C:\x\steamapps\common\x.exe") == "other"
    assert categorize("WindowsTerminal.exe") == "terminal"
    assert categorize("Zoom.exe") == "meeting"
    assert categorize("chrome.exe") == "browser"
    assert categorize("explorer.exe") == "other"


def act(t, category="other", running=(), games=(), fg="app.exe"):
    return Activity(t, fg, "", category, frozenset({category, *running}), tuple(games))


def feed(decider, samples, meeting_on=False):
    changes = []
    for a in samples:
        d = decider.update(a, a.time, meeting_on)
        if d:
            changes.append((a.time, d.mode))
    return changes


def test_gaming_switches_after_the_stable_time_and_back():
    d = ModeDecider(stable_s=20)
    game = [act(t, "game", games=["dota2.exe"]) for t in range(0, 40, 5)]
    assert feed(d, game) == [(20, "gaming")]
    assert "dota2.exe" in d.reason
    after = [act(t, "browser") for t in range(40, 80, 5)]
    assert feed(d, after) == [(60, "normal")]


def test_alt_tab_out_of_a_game_does_not_flip_the_mode():
    d = ModeDecider(stable_s=20)
    feed(d, [act(t, "game", games=["cs2.exe"]) for t in range(0, 30, 5)])
    assert d.mode == "gaming"
    # the game keeps running while a browser is in front
    assert feed(d, [act(t, "browser", running={"game"}, games=["cs2.exe"]) for t in range(30, 90, 5)]) == []


def test_coding_needs_half_of_the_last_minutes():
    d = ModeDecider(stable_s=10)
    mixed = [act(t, "ide" if (t // 5) % 3 else "browser") for t in range(0, 120, 5)]    # two thirds in the IDE
    assert d.mode == "normal" and feed(d, mixed)[0][1] == "coding"
    d2 = ModeDecider(stable_s=10)
    mostly_browser = [act(t, "ide" if (t // 5) % 3 == 0 else "browser") for t in range(0, 120, 5)]
    assert feed(d2, mostly_browser) == []


def test_meeting_from_calendar_is_immediate_and_beats_a_game():
    d = ModeDecider(stable_s=60)
    assert feed(d, [act(0, "game", games=["dota2.exe"])], meeting_on=True) == [(0, "meeting")]


def test_meeting_app_in_front_for_a_minute():
    d = ModeDecider(stable_s=0)
    assert feed(d, [act(t, "meeting", fg="Zoom.exe") for t in range(0, 70, 5)]) == [(60, "meeting")]


def test_voice_override_and_auto():
    d = ModeDecider(stable_s=20)
    feed(d, [act(t, "game", games=["dota2.exe"]) for t in range(0, 30, 5)])
    d.request("normal", now=30)
    assert feed(d, [act(30, "game", games=["dota2.exe"])]) == [(30, "normal")]
    assert feed(d, [act(t, "game", games=["dota2.exe"]) for t in range(35, 100, 5)]) == []   # held
    d.request("auto", now=100)
    assert feed(d, [act(t, "game", games=["dota2.exe"]) for t in range(100, 130, 5)]) == [(120, "gaming")]
    with pytest.raises(ValueError):
        d.request("party", now=0)


def test_voice_override_expires():
    d = ModeDecider(stable_s=0)
    d.request("coding", now=0, for_s=60)
    assert feed(d, [act(0)]) == [(0, "coding")]
    assert feed(d, [act(t) for t in range(5, 70, 5)]) == [(60, "normal")]


class FakeSystem:
    def __init__(self):
        self.notifications_on, self.plan, self.closed = True, "balanced-guid", []

    def notifications(self):
        return self.notifications_on

    def set_notifications(self, on):
        self.notifications_on = on

    def power_plan(self):
        return self.plan

    def set_power_plan(self, guid):
        self.plan = guid

    def close_app(self, name):
        self.closed.append(name)


def test_actions_apply_and_restore():
    system = FakeSystem()
    actions = ModeActions(system, {**DEFAULTS, "gaming": {**DEFAULTS["gaming"], "close": ["Code"]}})
    assert actions.switch("gaming") == ["notifications off", "power plan: high performance", "asked Code to close"]
    assert not system.notifications_on and system.plan == HIGH_PERFORMANCE
    actions.switch("meeting")                       # restores, then turns notifications off again
    assert not system.notifications_on and system.plan == "balanced-guid"
    actions.switch("normal")
    assert system.notifications_on and system.plan == "balanced-guid"


def test_settings_file_is_created_with_defaults(tmp_path):
    path = tmp_path / "modes.json"
    assert load_settings(path) == DEFAULTS and path.exists()
    path.write_text('{"gaming": {"notifications": false}}', encoding="utf-8")
    assert load_settings(path)["gaming"] == {"notifications": False} and load_settings(path)["coding"] == {}


def test_autopilot_end_to_end(tmp_path):
    clock = {"t": 1000.0}
    samples = {"next": act(1000, "browser")}
    system = FakeSystem()
    activity_log = ActivityLog()
    pilot = Autopilot(lambda: samples["next"], ModeActions(system, DEFAULTS), ModeDecider(stable_s=10),
                      activity_log, clock=lambda: clock["t"])
    with EventBus(tmp_path / "bus.sqlite3") as bus:
        bus.subscribe(NAME, ["calendar.event.*", "mode.requested"])
        bus.subscribe("watch", ["mode.changed", "assistant.reply"])

        def tick(dt=5.0, sample=None):
            clock["t"] += dt
            samples["next"] = sample or act(clock["t"], "browser")
            pilot.step(bus)

        tick()
        bus.publish(Event("calendar.event.soon", {"id": 1, "title": "Созвон", "kind": "meeting",
                                                  "minutes_to_start": 2}))
        tick()
        bus.publish(Event("calendar.event.started", {"id": 1, "kind": "meeting"}))
        tick()
        assert pilot.decider.mode == "meeting" and not system.notifications_on
        bus.publish(Event("calendar.event.ended", {"id": 1, "kind": "meeting"}))
        tick()
        assert pilot.decider.mode == "meeting"         # leaving waits for the stable time, like any switch
        tick()
        tick()
        assert pilot.decider.mode == "normal" and system.notifications_on
        bus.publish(Event("mode.requested", {"mode": "gaming", "language": "ru"}))
        tick()
        assert system.plan == HIGH_PERFORMANCE
        seen = [(e.topic, e.payload.get("mode") or e.payload.get("text")) for e in bus.poll("watch")]
    assert seen == [("assistant.reply", "Через 2 мин.: Созвон."), ("mode.changed", "meeting"),
                    ("mode.changed", "normal"), ("mode.changed", "gaming"),
                    ("assistant.reply", "Включён игровой режим.")]
    rows = activity_log.rows()
    assert len(rows) == 7 and rows[-1][1] == "gaming" and rows[-1][2] is True     # voice: a correction
