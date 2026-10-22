"""Microphone → wake word → one command → text on the bus.

The microphone is read in 80 ms frames. While idle, every frame goes to the wake-word detector (our own
CNN for «Амадеус», aiforme/wakeword) and updates the background noise level. After the wake word the frames
are recorded until the speaker stops — loudness falls back to the noise level for `silence_s` — and the
recording goes to Whisper. The result is published as `voice.transcribed`; the intent module takes it from
there. With `--push-to-talk` the Enter key replaces the wake word.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from aiforme.core import Event, EventBus

SAMPLE_RATE = 16_000
FRAME = 1280                    # 80 ms
FRAME_S = FRAME / SAMPLE_RATE

# Whisper's well-known inventions on silence and noise, mostly from subtitled videos in its training data
HALLUCINATIONS = {"продолжение следует", "субтитры сделал dimatorzok", "субтитры создавал dimatorzok",
                  "спасибо за просмотр", "редактор субтитров", "thank you for watching", "thanks for watching",
                  "thank you", "you", "bye"}


# --- end of speech ------------------------------------------------------------------------------------

@dataclass
class Endpointer:
    """Decides from frame loudness (RMS of int16 samples) when a command is over."""
    silence_s: float = 0.8      # quiet after speech that ends the command
    no_speech_s: float = 3.0    # give up if nothing is said after the wake word
    max_s: float = 15.0         # hard limit, for dictation
    ratio: float = 3.0          # speech is this many times louder than the background...
    min_rms: float = 300.0      # ...and at least this loud
    noise: float = 100.0        # background level, updated while idle

    def observe_noise(self, rms: float) -> None:
        self.noise = 0.95 * self.noise + 0.05 * rms

    def start(self) -> None:
        self._elapsed = self._quiet = 0     # in frames: summing 0.08 s floats drifts below the limits
        self._speech = False

    def push(self, rms: float) -> str | None:
        """Returns None while the command goes on, else why it ended: 'end', 'no_speech' or 'max'."""
        self._elapsed += 1
        if rms > max(self.noise * self.ratio, self.min_rms):
            self._speech, self._quiet = True, 0
        else:
            self._quiet += 1
        if self._speech and self._quiet >= _frames(self.silence_s):
            return "end"
        if not self._speech and self._elapsed >= _frames(self.no_speech_s):
            return "no_speech"
        if self._elapsed >= _frames(self.max_s):
            return "max"
        return None


def _frames(seconds: float) -> int:
    return max(1, round(seconds / FRAME_S))


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


# --- wake word ----------------------------------------------------------------------------------------

class WakeWord(Protocol):
    def detected(self, frame: np.ndarray) -> bool: ...
    def reset(self) -> None: ...


class EnterKey:
    """Push-to-talk from the console: press Enter, then speak."""

    def __init__(self) -> None:
        self._pressed = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()
        print("Press Enter, then say a command.", flush=True)

    def _read(self) -> None:
        for _ in sys.stdin:
            self._pressed.set()

    def detected(self, frame: np.ndarray) -> bool:
        if self._pressed.is_set():
            self._pressed.clear()
            return True
        return False

    def reset(self) -> None:
        self._pressed.clear()


# --- speech to text -----------------------------------------------------------------------------------

class WhisperSTT:
    """faster-whisper on the GPU when it works, else on the CPU with a smaller model."""

    def __init__(self, model: str = "large-v3-turbo", device: str = "auto", cpu_model: str = "small",
                 languages: tuple[str, ...] = ("ru", "en")):
        from faster_whisper import WhisperModel
        self.languages = languages
        if device in ("auto", "cuda"):
            _add_cuda_dlls()
            try:
                self._model = WhisperModel(model, device="cuda", compute_type="float16")
                self._model.transcribe(np.zeros(SAMPLE_RATE, np.float32), language="en")  # loads cuDNN now
                self.device, self.model_name = "cuda", model
                return
            except Exception as exc:
                if device == "cuda":
                    raise
                print(f"GPU unavailable for Whisper ({exc}); using the CPU", file=sys.stderr)
        self._model = WhisperModel(cpu_model, device="cpu", compute_type="int8")
        self.device, self.model_name = "cpu", cpu_model

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """int16 samples -> (text, language). The language is the likelier of `languages`: on a two-word
        command Whisper's free choice sometimes lands on Ukrainian or Bulgarian."""
        x = audio.astype(np.float32) / 32768.0
        _, _, probs = self._model.detect_language(x)
        lang = max(self.languages, key=lambda code: dict(probs).get(code, 0.0))
        segments, _ = self._model.transcribe(x, language=lang, beam_size=5, without_timestamps=True,
                                             condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip(), lang


def _add_cuda_dlls() -> None:
    """cuBLAS and cuDNN come as pip wheels (the `voice` extra); CTranslate2 finds them only on PATH."""
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    for root in (spec.submodule_search_locations if spec else []):
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


# --- the loop -----------------------------------------------------------------------------------------

class Listener:
    def __init__(self, bus: EventBus, wake: WakeWord, stt: WhisperSTT | Callable,
                 endpointer: Endpointer | None = None, beep: Callable[[int], None] | None = None,
                 input_device: int | str | None = None):
        self.bus, self.wake, self.stt = bus, wake, stt
        self.endpointer = endpointer or Endpointer()
        self.beep = beep or _beep
        self.input_device = input_device

    def run(self, stop: threading.Event) -> None:
        import sounddevice as sd
        frames: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, n, t, status):
            frames.put(indata[:, 0].copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
                            device=self.input_device, callback=callback):
            print("Listening. Say «Амадеус» and a command; Ctrl+C to stop.", flush=True)
            self.loop(frames, stop)

    def loop(self, frames: queue.Queue, stop: threading.Event) -> None:
        """The state machine on its own, so tests can feed it frames without a microphone."""
        recording: list[np.ndarray] | None = None
        while not stop.is_set():
            try:
                frame = frames.get(timeout=0.2)
            except queue.Empty:
                continue
            level = rms(frame)
            if recording is None:
                self.endpointer.observe_noise(level)
                if self.wake.detected(frame):
                    self.bus.publish(Event("voice.wake.detected", source="voice_listener"))
                    self.beep(880)
                    self.endpointer.start()
                    recording = []
                continue
            recording.append(frame)
            ended = self.endpointer.push(level)
            if ended is None:
                continue
            if ended == "no_speech":
                self.bus.publish(Event("voice.timeout", source="voice_listener"))
                self.beep(440)
            else:
                self._transcribe(np.concatenate(recording))
            recording = None
            _drain(frames)          # what was said during transcription is not a wake word
            self.wake.reset()

    def _transcribe(self, audio: np.ndarray) -> None:
        t0 = time.perf_counter()
        text, lang = self.stt.transcribe(audio)
        stt_ms = round((time.perf_counter() - t0) * 1000)
        if not text or text.lower().strip(" .!?…") in HALLUCINATIONS:
            self.bus.publish(Event("voice.timeout", {"heard": text}, source="voice_listener"))
            return
        print(f"you: {text}  [{lang}, {stt_ms} ms]", flush=True)
        self.bus.publish(Event("voice.transcribed", {
            "text": text, "language": lang, "audio_s": round(len(audio) / SAMPLE_RATE, 2), "stt_ms": stt_ms,
        }, source="voice_listener"))


def _drain(frames: queue.Queue) -> None:
    while True:
        try:
            frames.get_nowait()
        except queue.Empty:
            return


def _beep(freq: int) -> None:
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(freq, 120), daemon=True).start()
    except ImportError:
        pass
