"""Synthetic wake-word data from the voices built into Windows.

Three voices (Irina, Zira, Hazel) × three speaking rates × five pitches say the wake word, alone and followed by
a command; the same voices read every training command and the confusable words as negatives. Synthetic
speech is cheap variety in pitch and tempo, but it is not a person in a room — the evaluation in train.py
measures how much the recordings of a real voice add on top of it.

    python -m aiforme.wakeword.synth
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

from aiforme.core.config import data_dir
from aiforme.voice_orchestrator.nlu import TRAIN, language, load_examples
from aiforme.wakeword.record import CONFUSABLES

VOICES = {"Microsoft Irina Desktop": ("ru-RU", "Амадеус"),
          "Microsoft Zira Desktop": ("en-US", "Amadeus"),
          "Microsoft Hazel Desktop": ("en-GB", "Amadeus")}
# x-slow and x-fast are far outside human tempo (2.8 s and 0.3 s for a two-syllable word)
RATES = ["slow", "medium", "fast"]
PITCHES = ["x-low", "low", "medium", "high", "x-high"]

SYNTH = (
    "[Console]::InputEncoding=[Text.Encoding]::UTF8; Add-Type -AssemblyName System.Speech; "
    "$fmt=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
    "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono); "
    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "foreach($j in ([Console]::In.ReadToEnd() | ConvertFrom-Json)){"
    "$s.SelectVoice($j.voice); $s.SetOutputToWaveFile($j.path,$fmt); $s.SpeakSsml($j.ssml)}; $s.SetOutputToNull()"
)


def ssml(text: str, culture: str, rate: str = "medium", pitch: str = "medium") -> str:
    text = text.replace("&", "and").replace("<", "").replace(">", "")
    return (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{culture}'>"
            f"<prosody rate='{rate}' pitch='{pitch}'>{text}</prosody></speak>")


def jobs(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    root = data_dir() / "wakeword" / "synth"
    commands = {"ru": [], "en": []}
    for e in load_examples(TRAIN):
        commands[language(e.text)].append(e.text)
    out = []
    for voice, (culture, word) in VOICES.items():
        lang = culture[:2]
        for rate in RATES:
            for pitch in PITCHES:
                out.append({"kind": "positive", "voice": voice, "ssml": ssml(word, culture, rate, pitch)})
        for command in rng.sample(commands[lang], 20):     # the wake word as it is used: before a command
            out.append({"kind": "positive", "voice": voice,
                        "ssml": ssml(f"{word}, {command}", culture, rng.choice(RATES), rng.choice(PITCHES))})
        for text in commands[lang]:
            out.append({"kind": "negative", "voice": voice,
                        "ssml": ssml(text, culture, rng.choice(RATES), rng.choice(PITCHES))})
        for text in CONFUSABLES:
            for rate in RATES:
                out.append({"kind": "negative", "voice": voice, "ssml": ssml(text, culture, rate, rng.choice(PITCHES))})
    counts: dict[str, int] = {}
    for j in out:
        n = counts.get(j["kind"], 0)
        counts[j["kind"]] = n + 1
        j["path"] = str(root / j["kind"] / f"{n:04d}.wav")
    return out


def main() -> None:
    todo = jobs()
    for kind in ("positive", "negative"):
        Path(data_dir() / "wakeword" / "synth" / kind).mkdir(parents=True, exist_ok=True)
    subprocess.run(["powershell", "-NoProfile", "-Command", SYNTH], check=True,
                   input=json.dumps([{k: j[k] for k in ("path", "voice", "ssml")} for j in todo]).encode("utf-8"))
    print(f"{sum(j['kind'] == 'positive' for j in todo)} positive and "
          f"{sum(j['kind'] == 'negative' for j in todo)} negative clips in {data_dir() / 'wakeword' / 'synth'}")


if __name__ == "__main__":
    main()
