"""Demo data for the dashboard, so it can be seen without a microphone or weeks of use.

    python -m aiforme demo --dir ~/aiforme-demo                  # e5, as the assistant runs
    python -m aiforme demo --dir ~/aiforme-demo --embedder tfidf  # seconds, no model download
    python -m aiforme demo --dir ~/aiforme-demo --lang ru         # the commands in Russian, as I speak to it
    AIFORME_DATA_DIR=~/aiforme-demo python -m aiforme dashboard

Everything in it is made up — the calendar, the commands — but nothing is faked downstream of the commands: each one
goes through the real intent classifier, the labels go into the real feedback store, and the versions on the
Commands page come from the real gated retraining. Two rounds: the first commands are heard by version 0, labelled
and retrained on; the next ones by whatever version that produced, some labelled, some left for "Uncertain first".

The commands come in English (the default, so anyone can read the dashboard) or in Russian. Half of the intent
model's training phrases are English, so the English set is no easier for it.

It refuses a directory that already holds a bus, so it never mixes into real data.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

from aiforme.active_learning.retrain import current_model, retrain_and_announce
from aiforme.active_learning.store import Feedback
from aiforme.calendar import Calendar
from aiforme.core import Event, EventBus
from aiforme.voice_orchestrator.pipeline import IntentModule

# (text, the right intent, its slot value) — None: leave it unlabelled
Command = tuple[str, str | None, str | None]

# heard by version 0: everyday commands, slang the hand-written data lacks, and Whisper's slips
FIRST: list[Command] = [
    ("открой телеграм", "open_app", "телеграм"),
    ("запусти дискорд", "open_app", "дискорд"),
    ("раскочегарь дискорд", "open_app", "дискорд"),
    ("врубай телегу", "open_app", "телегу"),
    ("врубай спотик", "open_app", "спотик"),
    ("open visual studio code", "open_app", "visual studio code"),
    ("закрой хром", "close_app", "хром"),
    ("вырубай дискорд", "close_app", "дискорд"),
    ("close steam", "close_app", "steam"),
    ("найди файл с курсовой", "find_file", "курсовой"),
    ("погугли курс фунта", "web_search", "курс фунта"),
    ("search for amazon applied scientist internship", "web_search", "amazon applied scientist internship"),
    ("сделай погромче", "volume_up", None),
    ("потише", "volume_down", None),
    ("убери звук", "mute", None),
    ("поставь на паузу", "media_play_pause", None),
    ("продолжу видео", "media_play_pause", None),
    ("скипни трек", "media_next", None),
    ("сколько времени", "tell_time", None),
    ("what's on my calendar today", "agenda", None),
    ("что у меня сегодня", "agenda", None),
    ("игровой режим", "set_mode", "игровой"),
    ("заблокируй комп", "lock_screen", None),
    ("сделай скрин", "screenshot", None),
    ("отмена", "cancel", None),
    ("открой окно в комнате", "out_of_scope", None),
    ("какая завтра погода в винчестере", "web_search", "погода в винчестере"),
]

# heard by the version the first round produced
SECOND: list[Command] = [
    ("врубай телегу", "open_app", "телегу"),
    ("раскочегарь стим", "open_app", "стим"),
    ("вырубай телегу", "close_app", "телегу"),
    ("режим кодинга", "set_mode", "кодинга"),
    ("убавь звук", "volume_down", None),
    ("врубай музыку", "media_play_pause", None),
    ("напиши привет как дела", "dictate", "привет как дела"),
    ("switch to meeting mode", "set_mode", "meeting"),
    ("кинь трек дальше", None, None),
    ("покажи что в календаре", None, None),
    ("сделай потише и поставь на паузу", None, None),
    ("закинь в заметки купить молоко", None, None),
    ("включи свет", None, None),
    ("дай мне минутку", None, None),
]

# the same story in English: everyday commands, slang the hand-written data lacks, and phrasings it never saw
FIRST_EN: list[Command] = [
    ("open telegram", "open_app", "telegram"),
    ("launch discord", "open_app", "discord"),
    ("boot up discord", "open_app", "discord"),
    ("pull up spotify", "open_app", "spotify"),
    ("spin up telegram", "open_app", "telegram"),
    ("open visual studio code", "open_app", "visual studio code"),
    ("close chrome", "close_app", "chrome"),
    ("shut discord", "close_app", "discord"),
    ("close steam", "close_app", "steam"),
    ("find my coursework file", "find_file", "coursework"),
    ("google the pound exchange rate", "web_search", "the pound exchange rate"),
    ("search for amazon applied scientist internship", "web_search", "amazon applied scientist internship"),
    ("turn it up", "volume_up", None),
    ("bit quieter", "volume_down", None),
    ("kill the sound", "mute", None),
    ("pause it", "media_play_pause", None),
    ("resume the video", "media_play_pause", None),
    ("skip this song", "media_next", None),
    ("what time is it", "tell_time", None),
    ("what's on my calendar today", "agenda", None),
    ("what have i got today", "agenda", None),
    ("gaming mode", "set_mode", "gaming"),
    ("lock my pc", "lock_screen", None),
    ("grab a screenshot", "screenshot", None),
    ("never mind", "cancel", None),
    ("open the window in my room", "out_of_scope", None),
    ("what's the weather in winchester tomorrow", "web_search", "the weather in winchester tomorrow"),
]

SECOND_EN: list[Command] = [
    ("spin up telegram", "open_app", "telegram"),
    ("boot up steam", "open_app", "steam"),
    ("shut telegram", "close_app", "telegram"),
    ("coding mode", "set_mode", "coding"),
    ("turn it down a bit", "volume_down", None),
    ("put some music on", "media_play_pause", None),
    ("type hello how are you", "dictate", "hello how are you"),
    ("switch to meeting mode", "set_mode", "meeting"),
    ("next one please", None, None),
    ("show me my calendar", None, None),
    ("turn it down and pause", None, None),
    ("add buy milk to my notes", None, None),
    ("turn on the lights", None, None),
    ("give me a minute", None, None),
]

# (day of the week from Monday, start, end, title, kind)
WEEK = [
    (0, "10:00", "12:00", "Machine Learning lecture", "other"),
    (0, "17:00", "22:00", "Смена", "shift"),
    (1, "11:00", "11:30", "Стендап", "meeting"),
    (1, "14:00", "16:00", "Artificial Intelligence lab", "other"),
    (2, "09:00", "10:00", "Mock interview: ML breadth", "meeting"),
    (2, "17:00", "22:00", "Смена", "shift"),
    (3, "10:00", "12:00", "Computer Architectures", "other"),
    (3, "15:00", "15:45", "Placement: наставник", "meeting"),
    (4, "13:00", "15:00", "Vision agent: Qwen2.5-VL eval", "other"),
    (5, "12:00", "18:00", "Смена", "shift"),
]


WEEK_EN = [(d, a, b, {"Смена": "Shift", "Стендап": "Stand-up", "Placement: наставник": "Placement: mentor"}
            .get(title, title), kind) for d, a, b, title, kind in WEEK]

DEMOS = {"en": (FIRST_EN, SECOND_EN, WEEK_EN), "ru": (FIRST, SECOND, WEEK)}


class DemoExists(Exception):
    """The directory already holds a bus: it may be real data."""


def build(directory: str | Path, embedder: str = "tfidf", today: datetime | None = None, log=print,
          lang: str = "en") -> dict:
    if lang not in DEMOS:
        raise ValueError(f"unknown language {lang!r}: expected one of {tuple(DEMOS)}")
    first, second, week = DEMOS[lang]
    root = Path(directory).expanduser()
    if (root / "bus.sqlite3").exists():
        raise DemoExists(f"{root} already holds a bus; pick an empty directory")
    root.mkdir(parents=True, exist_ok=True)
    today = today or datetime.now()

    with Calendar(root / "calendar.sqlite3") as calendar:
        monday = datetime.combine((today - timedelta(days=today.weekday())).date(), time())
        for day, start, end, title, kind in week:
            calendar.add(title, monday + timedelta(days=day, hours=_h(start)),
                         monday + timedelta(days=day, hours=_h(end)), kind)

    summary = {"versions": []}
    with EventBus(root / "bus.sqlite3") as bus, Feedback(root / "feedback.sqlite3") as feedback:
        bus.publish(Event("mode.changed", {"mode": "coding", "previous": "normal", "actions": [],
                                           "reason": "an IDE or a terminal in front 80% of the last five minutes"},
                          source="mode_autopilot"))
        for round_no, commands in enumerate((first, second), start=1):
            classifier, version = current_model(feedback, embedder)
            log(f"round {round_no}: {len(commands)} commands heard by version {version} ({embedder})")
            module = IntentModule(classifier, version)
            for text, intent, slot in commands:
                heard = bus.publish(Event("voice.transcribed", {"text": text}, source="demo"))
                for out in module.handle(heard):
                    event = bus.publish(out)
                    if intent is not None and (round_no == 1 or len(feedback.labels(pending=True)) < 6):
                        p = event.payload
                        predicted = "out_of_scope" if event.topic == "intent.unknown" else p["intent"]
                        feedback.label(event.id, text, predicted, p.get("slots") or {}, p["confidence"],
                                       p["model"], event.created_at, intent, _slots(intent, slot))
            if round_no == 1:
                v = retrain_and_announce(feedback, bus, embedder)
                m = v.metrics
                log(f"  version {v.id} {'accepted' if v.accepted else 'rejected'}: test "
                    f"{m['test_before']['accuracy']:.1%} -> {m['test']['accuracy']:.1%}, labelled commands "
                    f"{m['real_before']['accuracy']:.1%} -> {m['real']['accuracy']:.1%} (out of fold)")
                summary["versions"].append({"id": v.id, "accepted": v.accepted})
        summary["pending"] = len(feedback.labels(pending=True))
    return summary


def _slots(intent: str, value: str | None) -> dict[str, str]:
    from aiforme.voice_orchestrator.intents import INTENTS
    name = INTENTS.get(intent)
    return {name: value} if name and value else {}


def _h(clock: str) -> float:
    hours, minutes = clock.split(":")
    return int(hours) + int(minutes) / 60
