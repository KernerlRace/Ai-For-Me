"""Wake-word recordings from someone else's Mac, packed into one zip to send back.

    bash record.sh                    # the menu: «Амадеус», other phrases, the room, or everything in turn
    bash record.sh --split test       # a separate session for the evaluation

A standalone copy of aiforme/wakeword/record.py: the person recording has no repository, only Python 3.9
from the Command Line Tools. Files keep the same layout and names — wakeword/<kind>/<split>/<speaker>_... —
so the zip unpacks straight into data/. The phrases are the training commands, copied into phrases.txt.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
WAKE_WORD = "Амадеус"
HERE = Path(__file__).resolve().parent
OUT = Path.home() / "AmadeusRecordings"

# the same as in record.py: how to say it, changing every 10 takes
STYLES = [
    "обычно", "тихо, почти шёпотом", "громко, как будто из другой комнаты", "быстро", "медленно, растягивая",
    "с вопросительной интонацией: «Амадеус?»", "отойди на 1–2 метра от микрофона", "устало или лениво",
    "сразу с командой: «Амадеус, открой браузер» — любую команду", "отвернувшись от микрофона",
]
CONFUSABLES = ["Амадей", "Моцарт Амадей", "Асмодей", "Мадейра", "Мадрид", "мадам", "Медуза", "Адам",
               "Амадина", "а мы же дома", "Ама", "Amadeo", "Amanda", "Madeira", "Medusa", "Matthew"]


def save(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype(np.int16).tobytes())


class Mic:
    """Records 16 kHz mono int16. Most Mac microphones open at 16 kHz directly; one that does not is recorded
    at its own rate and resampled through the FFT, which also cuts everything above 8 kHz."""

    def __init__(self, device=None):
        self.device = device
        try:
            sd.check_input_settings(device=device, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
            self.rate = SAMPLE_RATE
        except Exception:
            self.rate = int(sd.query_devices(device, "input")["default_samplerate"])

    def record(self, seconds: float) -> np.ndarray:
        audio = sd.rec(int(seconds * self.rate), samplerate=self.rate, channels=1, dtype="int16",
                       device=self.device)
        sd.wait()
        audio = audio[:, 0]
        if self.rate == SAMPLE_RATE:
            return audio
        n = int(round(len(audio) * SAMPLE_RATE / self.rate))
        spectrum = np.fft.rfft(audio.astype(np.float64))[: n // 2 + 1]
        out = np.fft.irfft(spectrum, n) * n / len(audio)
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)


def loud_enough(audio: np.ndarray) -> bool:
    """Speech somewhere in the clip: the loudest 50 ms is well above the quietest."""
    frames = audio[: len(audio) // 800 * 800].astype(np.float32).reshape(-1, 800)
    level = np.sqrt((frames ** 2).mean(axis=1))
    return level.max() > max(4 * np.percentile(level, 10), 300)


def beep() -> None:
    t = np.arange(int(0.12 * SAMPLE_RATE)) / SAMPLE_RATE
    try:
        sd.play((0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), SAMPLE_RATE)
        sd.wait()
    except Exception:                                  # no speakers: the «● говори» line is the signal
        pass


def check_mic(mic: Mic) -> bool:
    """macOS gives Terminal silence, not an error, until the microphone is allowed for it."""
    for attempt in range(2):
        input("Проверка микрофона: нажми Enter и скажи «раз, два, три» → ")
        audio = mic.record(3.0)
        if not np.any(audio):
            if attempt == 0:
                print("   Тишина. Если появилось окно «Терминал запрашивает доступ к микрофону» — нажми "
                      "«Разрешить» и попробуем ещё раз.\n")
                continue
            print("\nМикрофон не даёт звук. Открой «Системные настройки → Конфиденциальность и безопасность → "
                  "Микрофон», включи «Терминал», закрой Терминал полностью (Cmd+Q) и запусти скрипт заново.")
            return False
        if loud_enough(audio):
            print("   ✓ слышно хорошо\n")
            return True
        print("   Очень тихо — говори громче или ближе к микрофону.\n")
    return False


def takes(mic: Mic, kind: str, n: int, split: str, speaker: str) -> None:
    out = OUT / "wakeword" / kind / split
    stamp = time.strftime("%Y%m%d-%H%M%S")
    phrases = [p.strip() for p in (HERE / "phrases.txt").read_text(encoding="utf-8").splitlines() if p.strip()]
    phrases += CONFUSABLES * 3
    random.shuffle(phrases)
    what = f"«{WAKE_WORD}»" if kind == "positive" else "фразу с экрана"
    print(f"\nКак это работает — {n} записей, каждая одинаково:\n"
          f"  1. Читаешь задание.\n"
          f"  2. Нажимаешь Enter.\n"
          f"  3. Слышишь короткий писк — сразу говоришь {what}.\n"
          f"  4. Через пару секунд появится «✓ сохранено» и следующее задание.\n"
          f"Нужен перерыв — Ctrl+C: всё записанное сохранится.\n")
    i = 0
    while i < n:
        if kind == "positive":
            style = STYLES[(i // 10) % len(STYLES)]
            seconds = 3.0 if "команд" in style else 1.8
            prompt = f"Скажи «{WAKE_WORD}» — {style}"
        else:
            seconds, prompt = 3.0, f"Прочитай: «{phrases[i % len(phrases)]}»"
        input(f"[{i + 1}/{n}] {prompt}.\n   Нажми Enter → ")
        beep()
        time.sleep(0.15)                                   # let the beep fade
        print("   ● говори...", flush=True)
        audio = mic.record(seconds)
        if not loud_enough(audio):
            print("   Ничего не слышно — повторим это же задание.\n")
            continue
        name = f"{speaker}_{stamp}_{i:03d}.wav"
        save(out / name, audio)
        # per speaker, so unpacking into data/ does not overwrite the prompts of earlier sessions
        with open(out / f"prompts_{speaker}.tsv", "a", encoding="utf-8") as log:
            log.write(f"{name}\t{prompt}\n")
        i += 1
        print("   ✓ сохранено\n")
    print(f"Готово: {n} записей.")


def ambient(mic: Mic, minutes: float, split: str, speaker: str) -> None:
    """One file per minute, so a cut-off session keeps everything before the cut."""
    out = OUT / "wakeword" / "ambient" / split
    stamp = time.strftime("%Y%m%d-%H%M%S")
    total = int(np.ceil(minutes))
    ends = time.strftime("%H:%M", time.localtime(time.time() + minutes * 60))
    print(f"\nЗапись фона уже идёт, нажимать ничего не нужно. Длится {minutes:g} мин, закончится около {ends}.\n"
          f"Занимайся своими делами — хорошо, если играет видео, музыка, подкаст или идёт разговор. Только не "
          f"говори «{WAKE_WORD}». Крышку ноутбука не закрывай.\n"
          f"Ctrl+C — остановить раньше: сохранятся все полные минуты.\n")
    for m in range(total):
        save(out / f"{speaker}_{stamp}_{m:02d}.wav", mic.record(min(60.0, (minutes - m) * 60)))
        print(f"   записано {m + 1} из {total} мин", flush=True)
    print("Готово: фон записан.")


def pack(speaker: str) -> None:
    if not (OUT / "wakeword").exists():
        return
    zip_path = Path.home() / "Desktop" / f"amadeus_{speaker}"      # rebuilt after every part, always all of it
    made = shutil.make_archive(str(zip_path), "zip", root_dir=OUT, base_dir="wakeword")
    count = sum(1 for _ in (OUT / "wakeword").rglob("*.wav"))
    print(f"\n📦 Архив на рабочем столе: {Path(made).name} ({count} записей).\n"
          f"   Его и нужно отправить. Записи остаются и в {OUT}.")


def ask_speaker(default: str | None) -> str:
    while True:
        raw = default or input("Как тебя подписать? Одно слово латиницей (например anna): ")
        default = None
        name = re.sub(r"[^a-z0-9]", "", raw.lower())
        if name[:1].isalpha() and name != "me":        # «me» is the owner; a leading digit reads as «me» too
            return name
        print("   Нужно слово из латинских букв, начиная с буквы, и не «me».")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", default=None, help="a short Latin name: anna, kate, ...")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--positive", type=int, default=100, help="how many «Амадеус» takes")
    parser.add_argument("--negative", type=int, default=60, help="how many other phrases")
    parser.add_argument("--minutes", type=float, default=10, help="how long the room recording is")
    parser.add_argument("--device", default=None, help="microphone index or name (see: python -m sounddevice)")
    args = parser.parse_args()
    device = int(args.device) if (args.device or "").isdigit() else args.device

    print(f"\nПривет! Это запись голоса для ассистента «{WAKE_WORD}». Спасибо, что помогаешь 💛\n")
    speaker = ask_speaker(args.speaker)
    mic = Mic(device)
    try:
        if not check_mic(mic):
            sys.exit(1)
        parts = {
            "1": (f"«{WAKE_WORD}» {args.positive} раз (~{args.positive // 10} мин)",
                  lambda: takes(mic, "positive", args.positive, args.split, speaker)),
            "2": (f"другие фразы, {args.negative} шт. (~{args.negative // 10} мин)",
                  lambda: takes(mic, "negative", args.negative, args.split, speaker)),
            "3": (f"фон комнаты, {args.minutes:g} мин (говорить не нужно)",
                  lambda: ambient(mic, args.minutes, args.split, speaker)),
        }
        while True:
            print("Что записываем?")
            for key, (title, _) in parts.items():
                print(f"  {key} — {title}")
            print("  4 — всё по очереди: 1, 2, 3\n  0 — закончить и собрать архив")
            choice = input("Номер → ").strip()
            if choice == "0":
                break
            if choice not in ("1", "2", "3", "4"):
                print("   Нет такого номера.\n")
                continue
            for key in ("1", "2", "3") if choice == "4" else (choice,):
                parts[key][1]()
                pack(speaker)
                print()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    pack(speaker)
    print("Спасибо! 🙌")


if __name__ == "__main__":
    main()
