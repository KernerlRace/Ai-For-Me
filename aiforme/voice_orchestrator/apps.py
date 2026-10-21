"""Matching a spoken name ("телегу", "вс код", "photoshop") to an installed app, a process or a file name.

Speech gives Russian slang, inflected words and transliterations of English names; installed apps have English
names. Each spoken phrase is expanded into variants (aliases, Latin transliteration, the word without its case
ending) and compared with every candidate. Every contiguous span of up to three words is tried too, so a slot
that kept part of the command ("а можешь запустить стим") still finds Steam.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

from aiforme.voice_orchestrator.nlu import normalize

# spoken name -> what the app is called on Windows; the Russian side is matched after normalisation
ALIASES = {
    "хром": "chrome", "гугл хром": "chrome", "стим": "steam", "телега": "telegram", "телеграм": "telegram",
    "дискорд": "discord", "ворд": "word", "эксель": "excel", "блокнот": "notepad",
    "калькулятор": "calculator", "проводник": "file explorer", "настройки": "settings", "параметры": "settings",
    "вс код": "visual studio code", "вскод": "visual studio code", "vs code": "visual studio code",
    "vscode": "visual studio code", "обс": "obs studio", "obs": "obs studio", "дота": "dota 2", "dota": "dota 2",
    "диспетчер задач": "task manager", "ютуб": "youtube", "спотифай": "spotify", "фотошоп": "photoshop",
    "пейнт": "paint", "терминал": "terminal", "командная строка": "command prompt",
}

TRANSLIT = dict(zip("абвгдежзийклмнопрстуфхцчшщъыьэюя", [
    "a", "b", "v", "g", "d", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "f",
    "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"], strict=True))

MIN_SCORE = 0.72


def transliterate(text: str) -> str:
    return "".join(TRANSLIT.get(ch, ch) for ch in text)


def _stem(word: str) -> str:
    """Drops one case ending: дискорда → дискорд, телегу → телег. Crude, but enough for fuzzy matching."""
    return word[:-1] if len(word) > 4 and word[-1] in "ауеыиюя" else word


def variants(phrase: str) -> set[str]:
    phrase = normalize(phrase)
    stemmed = " ".join(_stem(w) for w in phrase.split())
    out = {phrase, stemmed}
    for p in (phrase, stemmed):
        for spoken, name in ALIASES.items():
            if p == spoken or _stem(p) == _stem(spoken):
                out.add(name)
    return {v for p in out for v in (p, transliterate(p)) if v}


def clean_name(name: str) -> str:
    """'Telegram Desktop®' -> 'telegram desktop'; 'WINWORD.EXE' -> 'winword'."""
    name = re.sub(r"\.exe$", "", name, flags=re.IGNORECASE)
    return normalize(re.sub(r"[®™©()]", " ", name))


def similarity(spoken: str, name: str) -> float:
    if not spoken or not name:
        return 0.0
    if spoken == name:
        return 1.0
    words = name.split()
    # a whole-word prefix of the name: "telegram" in "telegram desktop", "visual studio code" in "... insiders"
    for n in range(1, len(words) + 1):
        if spoken == " ".join(words[:n]):
            return 0.95 - 0.02 * (len(words) - n)
    if len(spoken) >= 4 and f" {spoken} " in f" {name} ":
        return 0.9
    best = SequenceMatcher(None, spoken, name).ratio()
    for n in range(1, len(words) + 1):
        best = max(best, SequenceMatcher(None, spoken, " ".join(words[:n])).ratio() - 0.03)
    return best


def spans(phrase: str, max_words: int = 3) -> list[str]:
    words = normalize(phrase).split()
    return [" ".join(words[i:i + n]) for n in range(min(max_words, len(words)), 0, -1)
            for i in range(len(words) - n + 1)]


def best_match(phrase: str, candidates: Iterable[str], min_score: float = MIN_SCORE) -> tuple[str, float] | None:
    """The candidate that best matches the phrase or one of its spans, with its score; None below min_score.
    Shorter spans pay a small penalty, so the whole phrase wins a tie against one of its words."""
    candidates = list(candidates)
    cleaned = [clean_name(c) for c in candidates]
    whole = normalize(phrase)
    best: tuple[str, float] | None = None
    for span in [whole, *[s for s in spans(phrase) if s != whole]]:
        penalty = 0.0 if span == whole else 0.04
        for spoken in variants(span):
            if len(spoken) < 3:
                continue
            for original, name in zip(candidates, cleaned, strict=True):
                score = similarity(spoken, name) - penalty
                if best is None or score > best[1]:
                    best = (original, score)
    return best if best and best[1] >= min_score else None
