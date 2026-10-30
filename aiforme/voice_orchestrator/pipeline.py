"""The voice orchestrator as bus modules: text → intent → action → spoken reply.

    voice.transcribed / text.command ─▶ voice_intent ─▶ intent.recognized / intent.unknown
    intent.model_updated ─▶ voice_intent (reloads the classifier a retraining accepted, aiforme/active_learning)
    intent.* ─▶ voice_actions ─▶ action.done / action.failed (+ assistant.reply, mode.requested)
    assistant.reply ─▶ voice_reply (prints, and speaks through Windows' built-in voices)

The listener (listener.py) publishes voice.transcribed; `python -m aiforme say` publishes text.command, so
everything after speech recognition can be tried without a microphone.
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from aiforme.core import Event, Module
from aiforme.voice_orchestrator.intents import INTENTS, OUT_OF_SCOPE
from aiforme.voice_orchestrator.nlu import IntentClassifier, language


class ActionError(Exception):
    """A command that was understood but could not be carried out. The English message goes to the log and
    to English replies; `ru` is said instead when the command was in Russian."""

    def __init__(self, message: str, ru: str | None = None):
        super().__init__(message)
        self.ru = ru or message


# A command is only meant for right now: a module's cursor survives restarts, and "lock the screen" said
# while the assistant was off must not fire when it starts an hour later.
MAX_AGE_S = 30.0


def _stale(event: Event) -> bool:
    return time.time() - event.created_at > MAX_AGE_S


class IntentModule(Module):
    """Every intent event names the model version behind it, so a label given in the dashboard later says which
    version was right or wrong. `reload` builds the current version again, for `intent.model_updated`."""
    name = "voice_intent"
    subscribes = ("voice.transcribed", "text.command", "intent.model_updated")

    def __init__(self, classifier: IntentClassifier, version: int = 0,
                 reload: Callable[[], tuple[IntentClassifier, int]] | None = None):
        self.classifier, self.version, self.reload = classifier, version, reload

    def handle(self, event: Event):
        if event.topic == "intent.model_updated":
            if self.reload:         # a few seconds; commands said meanwhile wait in the queue
                self.classifier, self.version = self.reload()
            return
        text = str(event.payload.get("text", "")).strip()
        if not text or _stale(event):
            return
        parse = self.classifier.predict_one(text)
        payload = {"text": text, "intent": parse.intent, "confidence": round(parse.confidence, 3),
                   "slots": parse.slots, "language": language(text), "model": self.version,
                   "threshold": self.classifier.threshold}
        if parse.intent == OUT_OF_SCOPE:
            yield Event("intent.unknown", {**payload, "best_guess": parse.best_guess})
        else:
            yield Event("intent.recognized", payload)


class Desktop(Protocol):
    def open_app(self, name: str) -> str: ...
    def close_app(self, name: str) -> str: ...
    def find_file(self, query: str) -> str: ...
    def web_search(self, query: str) -> str: ...
    def type_text(self, text: str) -> None: ...
    def press(self, key: str, times: int = 1) -> None: ...
    def screenshot(self) -> None: ...
    def media(self, action: str) -> str | None: ...
    def lock(self) -> None: ...


# (Russian, English)
REPLIES = {
    "unknown": ("Команда не распознана.", "Sorry, I didn't get that."),
    "no_slot": ("Не удалось разобрать, что именно.", "I didn't catch what exactly."),
    "failed": ("Не получилось: {error}.", "That didn't work: {error}."),
    "time": ("Сейчас {time}.", "It's {time}."),
    "agenda": ("Сегодня: {items}.", "Today: {items}."),
    "agenda_empty": ("Сегодня в календаре ничего нет.", "Nothing in the calendar today."),
    "agenda_none": ("Календарь не подключён.", "The calendar is not connected."),
    "nothing_playing": ("Сейчас ничего не играет.", "Nothing is playing."),
    "already_playing": ("Уже играет.", "It is already playing."),
}

# which way a play/pause command goes; resuming words are checked first: «сними с паузы» and "unpause" hold «пауз»
RESUME_WORDS = ("продолж", "сними", "убери пауз", "включи", "запусти", "play", "resume", "continue", "unpause")
PAUSE_WORDS = ("пауз", "стоп", "останов", "выключи", "замри", "pause", "stop", "hold")


def media_action(text: str) -> str:
    """«поставь видео на паузу» -> 'pause', «продолжи» -> 'play', «плей-пауза» or nothing clear -> 'toggle'."""
    t = text.lower()
    if any(w in t for w in RESUME_WORDS):
        return "play"
    if any(w in t for w in PAUSE_WORDS):
        return "pause"
    return "toggle"

# spoken mode name -> the autopilot's mode; matched on the start of each word, so cases and forms all work
MODE_WORDS = {
    "gaming": ("игр", "game", "gaming", "play"),
    "coding": ("код", "кодин", "работ", "рабоч", "програм", "coding", "code", "work", "dev"),
    "meeting": ("встреч", "созвон", "совещ", "звонк", "meeting", "call", "conference"),
    "normal": ("обычн", "нормал", "normal", "default", "usual"),
    "auto": ("авто", "auto", "automatic"),
}


def mode_of(phrase: str) -> str | None:
    """'игровой' / 'режим игры' / 'gaming' -> 'gaming'; None if no mode is named."""
    for word in phrase.lower().split():
        for mode, stems in MODE_WORDS.items():
            if any(word.startswith(stem) for stem in stems):
                return mode
    return None

VOLUME_STEP = 5     # each volume key press is 2%


class ActionModule(Module):
    name = "voice_actions"
    subscribes = ("intent.recognized", "intent.unknown")

    def __init__(self, desktop: Desktop, now: Callable[[], datetime] = datetime.now,
                 agenda: Callable[[datetime], list] | None = None):
        self.desktop, self.now, self.agenda = desktop, now, agenda

    def handle(self, event: Event):
        if _stale(event):
            return
        p = event.payload
        lang = 0 if p.get("language") == "ru" else 1
        if event.topic == "intent.unknown":
            yield _reply(REPLIES["unknown"][lang])
            return
        intent, slots = p["intent"], p.get("slots", {})
        slot = INTENTS.get(intent)
        value = slots.get(slot, "") if slot else ""
        if slot and not value:
            yield Event("action.failed", {"intent": intent, "error": f"no {slot}"})
            yield _reply(REPLIES["no_slot"][lang])
            return
        if intent == "set_mode":                     # the autopilot switches and confirms
            mode = mode_of(value)
            if mode is None:
                yield Event("action.failed", {"intent": intent, "slots": slots, "error": f"no mode in {value!r}"})
                yield _reply(REPLIES["no_slot"][lang])
                return
            yield Event("mode.requested", {"mode": mode, "language": "ru" if lang == 0 else "en"})
            yield Event("action.done", {"intent": intent, "slots": slots, "result": mode})
            return
        try:
            result, reply = self._do(intent, value, lang, str(p.get("text", "")))
        except ActionError as exc:
            yield Event("action.failed", {"intent": intent, "slots": slots, "error": str(exc)})
            yield _reply(REPLIES["failed"][lang].format(error=exc.ru if lang == 0 else exc))
            return
        yield Event("action.done", {"intent": intent, "slots": slots, "result": result})
        if reply:
            yield _reply(reply)

    def _do(self, intent: str, value: str, lang: int, text: str = "") -> tuple[str, str | None]:
        d = self.desktop
        match intent:
            case "open_app":
                return d.open_app(value), None
            case "close_app":
                return d.close_app(value), None
            case "find_file":
                return d.find_file(value), None
            case "web_search":
                return d.web_search(value), None
            case "dictate":
                d.type_text(value)
                return value, None
            case "volume_up" | "volume_down":
                d.press(intent, VOLUME_STEP)
                return intent, None
            case "mute":
                d.press("volume_mute")
                return intent, None
            case "media_play_pause":
                action = media_action(text)
                title = d.media(action)
                if title is None:
                    key = "already_playing" if action == "play" else "nothing_playing"
                    return action, REPLIES[key][lang]
                return f"{action}: {title}", None
            case "media_next":
                title = d.media("next")
                return (f"next: {title}", None) if title else ("next", REPLIES["nothing_playing"][lang])
            case "screenshot":
                d.screenshot()
                return intent, None
            case "lock_screen":
                d.lock()
                return intent, None
            case "tell_time":
                now = self.now().strftime("%H:%M")
                return now, REPLIES["time"][lang].format(time=now)
            case "cancel":
                return intent, None
            case "agenda":
                return intent, self._agenda(lang)
        raise ActionError(f"no action for intent {intent!r}", f"нет действия для «{intent}»")

    def _agenda(self, lang: int) -> str:
        if self.agenda is None:
            return REPLIES["agenda_none"][lang]
        events = self.agenda(self.now())
        if not events:
            return REPLIES["agenda_empty"][lang]
        items = ", ".join(f"{e.start:%H:%M} {e.title}" for e in events)
        return REPLIES["agenda"][lang].format(items=items)


def _reply(text: str) -> Event:
    return Event("assistant.reply", {"text": text})


class ReplyModule(Module):
    """Prints and speaks replies; completed actions are printed only, so silent ones still show up."""
    name = "voice_reply"
    subscribes = ("assistant.reply", "action.done", "mode.changed")

    def __init__(self, speak: Callable[[str], None] | None = None):
        self.speak = speak

    def handle(self, event: Event):
        if event.topic == "mode.changed":
            p = event.payload
            done = f" — {', '.join(p['actions'])}" if p.get("actions") else ""
            print(f"  ◆ mode: {p.get('mode')} ({p.get('reason')}){done}", flush=True)
            return
        if event.topic == "action.done":
            print(f"  ✓ {event.payload.get('intent')}: {event.payload.get('result')}", flush=True)
            return
        text = str(event.payload.get("text", ""))
        print(f"assistant: {text}", flush=True)
        if self.speak and text:
            self.speak(text)


SAPI = (
    "[Console]::InputEncoding=[Text.Encoding]::UTF8; Add-Type -AssemblyName System.Speech; "
    "$t=[Console]::In.ReadToEnd(); $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$c=if($t -match '[а-яА-ЯёЁ]'){'ru-RU'}else{'en-US'}; "
    "try{$s.SelectVoiceByHints('NotSet','NotSet',0,[Globalization.CultureInfo]$c)}catch{}; $s.Speak($t)"
)


def windows_speak(text: str) -> None:
    """Speaks without blocking through System.Speech; uses a Russian voice for Cyrillic text if one is
    installed (Settings → Time & language → Speech), otherwise the default voice."""
    proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", SAPI], stdin=subprocess.PIPE,
                            creationflags=0x08000000)
    proc.stdin.write(text.encode("utf-8"))
    proc.stdin.close()
