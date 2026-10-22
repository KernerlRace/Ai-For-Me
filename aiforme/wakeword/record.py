"""Recording the wake-word data with your own microphone.

    python -m aiforme.wakeword.record positive --n 200            # «Амадеус», in different ways
    python -m aiforme.wakeword.record negative --n 100            # other phrases, including similar words
    python -m aiforme.wakeword.record ambient --minutes 10         # the room as it is: talk, music, video
    ... --split test                                              # a separate session for the evaluation
    ... --speaker anna                                            # someone else's voice, scored separately

Files go to data/wakeword/<kind>/<split>/. The test split should be a different session — another day,
another distance from the microphone — so the evaluation does not reward remembering one sitting.
"""
from __future__ import annotations

import argparse
import random
import time
import wave
from pathlib import Path

import numpy as np

from aiforme.core.config import data_dir
from aiforme.voice_orchestrator.nlu import TRAIN, load_examples
from aiforme.wakeword import WAKE_WORD
from aiforme.wakeword.features import SAMPLE_RATE

# how to say it, changing every 10 takes: one voice said the same way 200 times generalises badly
STYLES = [
    "обычно", "тихо, почти шёпотом", "громко, как будто из другой комнаты", "быстро", "медленно, растягивая",
    "с вопросительной интонацией: «Амадеус?»", "отойдите на 1–2 метра от микрофона", "устало или лениво",
    "сразу с командой: «Амадеус, открой браузер» — любую команду", "отвернувшись от микрофона",
]
# words that sound close to the wake word: the model must learn they are not it
CONFUSABLES = ["Амадей", "Моцарт Амадей", "Асмодей", "Мадейра", "Мадрид", "мадам", "Медуза", "Адам",
               "Амадина", "а мы же дома", "Ама", "Амадеуса нет", "Amadeo", "Amanda", "Madeira", "Medusa",
               "Asmodeus", "Matthew"]


def save(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype(np.int16).tobytes())


def record(seconds: float, device=None) -> np.ndarray:
    import sounddevice as sd
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device)
    sd.wait()
    return audio[:, 0]


def loud_enough(audio: np.ndarray) -> bool:
    """Speech somewhere in the clip: the loudest 50 ms is well above the quietest."""
    frames = audio[: len(audio) // 800 * 800].astype(np.float32).reshape(-1, 800)
    level = np.sqrt((frames ** 2).mean(axis=1))
    return level.max() > max(4 * np.percentile(level, 10), 300)


def beep() -> None:
    try:
        import winsound
        winsound.Beep(880, 120)
    except ImportError:
        pass


def takes(kind: str, n: int, split: str, device, speaker: str) -> None:
    out = data_dir() / "wakeword" / kind / split
    stamp = time.strftime("%Y%m%d-%H%M%S")
    phrases = [e.text for e in load_examples(TRAIN)] + CONFUSABLES * 3
    random.shuffle(phrases)
    what = f"«{WAKE_WORD}»" if kind == "positive" else "фразу с экрана"
    print(f"\nКак это работает — {n} записей, каждая одинаково:\n"
          f"  1. Читаете задание.\n"
          f"  2. Нажимаете Enter.\n"
          f"  3. Слышите короткий писк — сразу говорите {what}.\n"
          f"  4. Через пару секунд появится «✓ сохранено» и следующее задание.\n"
          f"После {n}-й записи программа сама напишет «Готово» и закроется.\n"
          f"Устали — Ctrl+C: всё записанное сохранится, потом можно запустить ещё раз.\n"
          f"Файлы: {out}\n")
    i = 0
    while i < n:
        if kind == "positive":
            style = STYLES[(i // 10) % len(STYLES)]
            seconds = 3.0 if "команд" in style else 1.8
            prompt = f"Скажите «{WAKE_WORD}» — {style}"
        else:
            seconds, prompt = 3.0, f"Прочитайте: «{phrases[i % len(phrases)]}»"
        input(f"[{i + 1}/{n}] {prompt}.\n   Нажмите Enter → ")
        beep()
        time.sleep(0.15)                                   # let the beep fade
        print("   ● говорите...", flush=True)
        audio = record(seconds, device)
        if not loud_enough(audio):
            print("   Ничего не слышно — повторим это же задание.\n")
            continue
        save(out / f"{speaker}_{stamp}_{i:03d}.wav", audio)
        i += 1
        print("   ✓ сохранено\n")
    print(f"Готово: {n} записей в {out}")


def ambient(minutes: float, split: str, device, speaker: str) -> None:
    """One file per minute, so a cut-off session keeps everything before the cut."""
    out = data_dir() / "wakeword" / "ambient" / split
    stamp = time.strftime("%Y%m%d-%H%M%S")
    total = int(np.ceil(minutes))
    print(f"\nЗапись фона уже идёт, нажимать ничего не нужно. Длится {minutes:g} мин: каждую минуту появится "
          f"строчка с прогрессом, в конце — «Готово».\nЗанимайтесь своими делами — хорошо, если играет видео, "
          f"подкаст или идёт разговор. Только не говорите «{WAKE_WORD}». Ctrl+C — остановить раньше.\n")
    for m in range(total):
        save(out / f"{speaker}_{stamp}_{m:02d}.wav", record(min(60.0, (minutes - m) * 60), device))
        print(f"   записано {m + 1} из {total} мин", flush=True)
    print(f"Готово: фон записан в {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("positive", "negative", "ambient"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--device", default=None, help="microphone index or name")
    parser.add_argument("--speaker", default="me", help="who is speaking, a short Latin name: me, anna, ...")
    args = parser.parse_args()
    device = int(args.device) if (args.device or "").isdigit() else args.device
    try:
        if args.kind == "ambient":
            ambient(args.minutes, args.split, device, args.speaker)
        else:
            takes(args.kind, args.n, args.split, device, args.speaker)
    except KeyboardInterrupt:
        print("\nОстановлено.")


if __name__ == "__main__":
    main()
