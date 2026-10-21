"""End to end on synthesized speech: test phrase → Windows text-to-speech → Whisper → intent.

    python -m aiforme.voice_orchestrator.spoken_eval --embedder minilm

Real voices are the real test, but recordings of people saying the 160 test commands do not exist yet.
Synthesized speech is clean and has no accent or room noise, so this measures what Whisper's mistakes cost
the intent classifier, not how the assistant copes with a real microphone. Russian phrases are read by
Microsoft Irina, English ones alternately by Zira (US) and Hazel (UK).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from aiforme.core.config import data_dir
from aiforme.voice_orchestrator.evaluate import RESULTS
from aiforme.voice_orchestrator.intents import INTENTS, OUT_OF_SCOPE
from aiforme.voice_orchestrator.listener import SAMPLE_RATE, WhisperSTT
from aiforme.voice_orchestrator.nlu import TEST, language, load_examples, normalize, train

VOICES = {"ru": ["Microsoft Irina Desktop"], "en": ["Microsoft Zira Desktop", "Microsoft Hazel Desktop"]}

SYNTH = (
    "[Console]::InputEncoding=[Text.Encoding]::UTF8; Add-Type -AssemblyName System.Speech; "
    "$fmt=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
    "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono); "
    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "foreach($j in ([Console]::In.ReadToEnd() | ConvertFrom-Json)){"
    "$s.SelectVoice($j.voice); $s.SetOutputToWaveFile($j.path,$fmt); $s.Speak($j.text)}; $s.SetOutputToNull()"
)


def synthesize(texts: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, counters = [], {"ru": 0, "en": 0}
    for i, text in enumerate(texts):
        lang = language(text)
        voice = VOICES[lang][counters[lang] % len(VOICES[lang])]
        counters[lang] += 1
        jobs.append({"path": str(out_dir / f"{i:03d}.wav"), "text": text, "voice": voice})
    subprocess.run(["powershell", "-NoProfile", "-Command", SYNTH], input=json.dumps(jobs).encode("utf-8"),
                   check=True)
    return [Path(j["path"]) for j in jobs]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="minilm")
    parser.add_argument("--stt-device", default="auto")
    args = parser.parse_args()

    test = load_examples(TEST)
    paths = synthesize([e.text for e in test], data_dir() / "tts_test")
    stt = WhisperSTT(device=args.stt_device)
    clf = train(args.embedder)

    rows = []
    for e, path in zip(test, paths, strict=True):
        heard, lang = stt.transcribe(read_wav(path))
        on_text, on_speech = clf.predict_one(e.text), clf.predict_one(heard)
        slot = INTENTS.get(e.intent)
        rows.append({
            "text": e.text, "heard": heard, "language": lang, "gold": e.intent,
            "transcript_exact": normalize(heard) == normalize(e.text),
            "intent_on_text": on_text.intent == e.intent,
            "intent_on_speech": on_speech.intent == e.intent,
            "pred_on_speech": on_speech.intent,
            "slot_on_speech": None if not slot else (
                on_speech.intent == e.intent and on_speech.slots.get(slot, "") == normalize(e.slots[slot])),
        })

    def rate(key, subset=rows):
        vals = [r[key] for r in subset if r[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    in_scope = [r for r in rows if r["gold"] != OUT_OF_SCOPE]
    report = {
        "embedder": args.embedder, "whisper": f"{stt.model_name} on {stt.device}", "phrases": len(rows),
        "transcript_exact": rate("transcript_exact"),
        "intent_accuracy_on_text": rate("intent_on_text"), "intent_accuracy_on_speech": rate("intent_on_speech"),
        "in_scope_on_text": rate("intent_on_text", in_scope), "in_scope_on_speech": rate("intent_on_speech", in_scope),
        "slot_end_to_end_on_speech": rate("slot_on_speech"),
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"spoken_{args.embedder}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=1))
    for r in rows:
        if not r["transcript_exact"]:
            print(f"  {r['text']!r} -> {r['heard']!r}" + ("" if r["intent_on_speech"] else
                                                         f"  [intent {r['pred_on_speech']}]"))


if __name__ == "__main__":
    main()
