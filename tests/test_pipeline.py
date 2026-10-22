import time
from dataclasses import replace
from datetime import datetime

import pytest

from aiforme.core import Event
from aiforme.voice_orchestrator.nlu import TRAIN, IntentClassifier, TfidfEmbedder, load_examples
from aiforme.voice_orchestrator.pipeline import MAX_AGE_S, ActionError, ActionModule, IntentModule, ReplyModule


class FakeDesktop:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))
            if args and args[0] == "photoshop":
                raise ActionError("no installed app matches 'photoshop'", "не нашлось приложения «photoshop»")
            return args[0] if args else None
        return record


@pytest.fixture(scope="module")
def intents():
    return IntentModule(IntentClassifier(TfidfEmbedder()).fit(load_examples(TRAIN)))


def run_actions(payload, topic="intent.recognized"):
    desktop = FakeDesktop()
    module = ActionModule(desktop, now=lambda: datetime(2026, 10, 21, 19, 5))
    return desktop.calls, list(module.handle(Event(topic, payload)))


def test_text_becomes_an_intent(intents):
    [out] = intents.handle(Event("voice.transcribed", {"text": "Открой Telegram."}))
    assert out.topic == "intent.recognized"
    assert out.payload["intent"] == "open_app"
    assert out.payload["slots"] == {"app": "telegram"}
    assert out.payload["language"] == "ru"


def test_out_of_scope_becomes_unknown(intents):
    [out] = intents.handle(Event("text.command", {"text": "order a pizza"}))
    assert out.topic == "intent.unknown"


def test_old_commands_are_ignored(intents):
    old = replace(Event("text.command", {"text": "lock the computer"}), created_at=time.time() - MAX_AGE_S - 1)
    assert list(intents.handle(old)) == []
    assert run_actions({"intent": "lock_screen"})[0] == [("lock",)]          # fresh: runs
    desktop = FakeDesktop()
    stale = replace(Event("intent.recognized", {"intent": "lock_screen"}), created_at=time.time() - 60)
    assert list(ActionModule(desktop).handle(stale)) == []
    assert desktop.calls == []


def test_empty_text_is_ignored(intents):
    assert list(intents.handle(Event("voice.transcribed", {"text": "  "}))) == []


def test_actions_call_the_desktop():
    calls, out = run_actions({"intent": "open_app", "slots": {"app": "telegram"}, "language": "ru"})
    assert calls == [("open_app", "telegram")]
    assert [e.topic for e in out] == ["action.done"]

    calls, _ = run_actions({"intent": "volume_up"})
    assert calls == [("press", "volume_up", 5)]
    calls, _ = run_actions({"intent": "mute"})
    assert calls == [("press", "volume_mute")]
    calls, _ = run_actions({"intent": "dictate", "slots": {"text": "привет"}})
    assert calls == [("type_text", "привет")]


def test_time_is_spoken_in_the_language_of_the_command():
    _, out = run_actions({"intent": "tell_time", "language": "ru"})
    assert out[-1].topic == "assistant.reply"
    assert out[-1].payload["text"] == "Сейчас 19:05."
    _, out = run_actions({"intent": "tell_time", "language": "en"})
    assert out[-1].payload["text"] == "It's 19:05."


def test_failures_are_reported_not_raised():
    _, out = run_actions({"intent": "open_app", "slots": {"app": "photoshop"}, "language": "en"})
    assert [e.topic for e in out] == ["action.failed", "assistant.reply"]
    assert "photoshop" in out[1].payload["text"]
    _, out = run_actions({"intent": "open_app", "slots": {"app": "photoshop"}, "language": "ru"})
    assert out[1].payload["text"] == "Не получилось: не нашлось приложения «photoshop»."

    calls, out = run_actions({"intent": "open_app", "slots": {}, "language": "ru"})
    assert calls == []
    assert [e.topic for e in out] == ["action.failed", "assistant.reply"]


def test_unknown_gets_a_reply():
    _, out = run_actions({"text": "order a pizza", "language": "en"}, "intent.unknown")
    assert [e.topic for e in out] == ["assistant.reply"]


def test_reply_is_printed_and_spoken(capsys):
    spoken = []
    ReplyModule(spoken.append).handle(Event("assistant.reply", {"text": "It's 19:05."}))
    assert spoken == ["It's 19:05."]
    assert "It's 19:05." in capsys.readouterr().out
    ReplyModule(spoken.append).handle(Event("action.done", {"intent": "open_app", "result": "Калькулятор"}))
    assert spoken == ["It's 19:05."]                       # printed, not spoken
    assert "open_app: Калькулятор" in capsys.readouterr().out


def test_mode_names():
    from aiforme.voice_orchestrator.pipeline import mode_of
    assert mode_of("игровой") == mode_of("режим игры") == mode_of("Gaming") == "gaming"
    assert mode_of("кодинга") == mode_of("рабочий") == mode_of("work") == "coding"
    assert mode_of("созвона") == mode_of("встречи") == "meeting"
    assert mode_of("обычный") == "normal" and mode_of("автоматический") == "auto"
    assert mode_of("весёлый") is None


def test_set_mode_asks_the_autopilot():
    _, out = run_actions({"intent": "set_mode", "slots": {"mode": "игровой"}, "language": "ru"})
    assert [(e.topic, e.payload.get("mode")) for e in out] == [("mode.requested", "gaming"), ("action.done", None)]
    assert out[0].payload["language"] == "ru"
    _, out = run_actions({"intent": "set_mode", "slots": {"mode": "весёлый"}, "language": "ru"})
    assert [e.topic for e in out] == ["action.failed", "assistant.reply"]


def test_agenda_reads_today():
    from types import SimpleNamespace
    today = [SimpleNamespace(start=datetime(2026, 10, 21, 10, 0), title="Созвон"),
             SimpleNamespace(start=datetime(2026, 10, 21, 12, 0), title="Смена")]
    module = ActionModule(FakeDesktop(), now=lambda: datetime(2026, 10, 21, 9, 0), agenda=lambda now: today)
    out = list(module.handle(Event("intent.recognized", {"intent": "agenda", "language": "ru"})))
    assert out[-1].payload["text"] == "Сегодня: 10:00 Созвон, 12:00 Смена."
    empty = ActionModule(FakeDesktop(), agenda=lambda now: [])
    out = list(empty.handle(Event("intent.recognized", {"intent": "agenda", "language": "en"})))
    assert out[-1].payload["text"] == "Nothing in the calendar today."


def test_media_commands_say_which_way():
    from aiforme.voice_orchestrator.pipeline import media_action
    assert media_action("Поставь паузу на видео.") == media_action("стоп видео") == "pause"
    assert media_action("сними с паузы") == media_action("продолжи") == media_action("unpause") == "play"
    assert media_action("плей") == "toggle"


def test_pause_acts_on_what_is_playing():
    class Media(FakeDesktop):
        def __init__(self, playing):
            super().__init__()
            self.playing = playing

        def media(self, action):
            self.calls.append(("media", action))
            if action == "pause" and not self.playing:
                return None
            return "Врата Штейна"

    module = ActionModule(Media(playing=True))
    out = list(module.handle(Event("intent.recognized", {"intent": "media_play_pause", "text": "пауза",
                                                         "language": "ru"})))
    assert module.desktop.calls == [("media", "pause")]
    assert out[0].payload["result"] == "pause: Врата Штейна" and len(out) == 1
    idle = ActionModule(Media(playing=False))
    out = list(idle.handle(Event("intent.recognized", {"intent": "media_play_pause", "text": "пауза",
                                                       "language": "ru"})))
    assert out[-1].payload["text"] == "Сейчас ничего не играет."
