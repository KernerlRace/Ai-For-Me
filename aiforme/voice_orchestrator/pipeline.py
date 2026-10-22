"""The voice orchestrator as bus modules: text → intent → action → spoken reply.

    voice.transcribed / text.command ─▶ voice_intent ─▶ intent.recognized / intent.unknown
    intent.* ─▶ voice_actions ─▶ action.done / action.failed (+ assistant.reply)
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
    name = "voice_intent"
    subscribes = ("voice.transcribed", "text.command")

    def __init__(self, classifier: IntentClassifier):
        self.classifier = classifier

    def handle(self, event: Event):
        text = str(event.payload.get("text", "")).strip()
        if not text or _stale(event):
            return
        parse = self.classifier.predict_one(text)
        payload = {"text": text, "intent": parse.intent, "confidence": round(parse.confidence, 3),
                   "slots": parse.slots, "language": language(text)}
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
    def lock(self) -> None: ...


# (Russian, English)
REPLIES = {
    "unknown": ("Команда не распознана.", "Sorry, I didn't get that."),
    "no_slot": ("Не удалось разобрать, что именно.", "I didn't catch what exactly."),
    "failed": ("Не получилось: {error}.", "That didn't work: {error}."),
    "time": ("Сейчас {time}.", "It's {time}."),
}

VOLUME_STEP = 5     # each volume key press is 2%


class ActionModule(Module):
    name = "voice_actions"
    subscribes = ("intent.recognized", "intent.unknown")

    def __init__(self, desktop: Desktop, now: Callable[[], datetime] = datetime.now):
        self.desktop, self.now = desktop, now

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
        try:
            result, reply = self._do(intent, value, lang)
        except ActionError as exc:
            yield Event("action.failed", {"intent": intent, "slots": slots, "error": str(exc)})
            yield _reply(REPLIES["failed"][lang].format(error=exc.ru if lang == 0 else exc))
            return
        yield Event("action.done", {"intent": intent, "slots": slots, "result": result})
        if reply:
            yield _reply(reply)

    def _do(self, intent: str, value: str, lang: int) -> tuple[str, str | None]:
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
            case "media_play_pause" | "media_next":
                d.press(intent)
                return intent, None
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
        raise ActionError(f"no action for intent {intent!r}", f"нет действия для «{intent}»")


def _reply(text: str) -> Event:
    return Event("assistant.reply", {"text": text})


class ReplyModule(Module):
    """Prints and speaks replies; completed actions are printed only, so silent ones still show up."""
    name = "voice_reply"
    subscribes = ("assistant.reply", "action.done")

    def __init__(self, speak: Callable[[str], None] | None = None):
        self.speak = speak

    def handle(self, event: Event):
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
