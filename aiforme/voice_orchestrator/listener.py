"""Microphone → wake word → one command → text on the bus.

The microphone is read in 80 ms frames. While idle, every frame goes to the wake-word detector (our own
CNN for «Амадеус», aiforme/wakeword) and updates the background noise level. After the wake word the frames
are recorded, with the computer's volume turned down, until the speaker stops — Silero VAD hears no speech for
`silence_s` — and the recording goes to Whisper. The result is published as `voice.transcribed`; the intent
module takes it from there. With `--push-to-talk` the Enter key replaces the wake word.
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
    """Decides when a command is over: from a speech detector's verdict per frame (`step`), or, without one,
    from frame loudness (`push`, RMS of int16 samples) against the background level."""
    silence_s: float = 0.9      # quiet after speech that ends the command
    no_speech_s: float = 3.0    # give up if nothing is said after the wake word
    max_s: float = 15.0         # hard limit, for dictation
    ratio: float = 3.0          # speech is this many times louder than the background...
    min_rms: float = 60.0       # ...and at least this loud. Low on purpose: with a noise-suppressing
                                # microphone silence is ~1 and a third of quiet speech is under 300
    noise: float = 100.0        # background level, updated while idle

    def observe_noise(self, rms: float) -> None:
        self.noise = 0.95 * self.noise + 0.05 * rms

    def start(self) -> None:
        self._elapsed = self._quiet = 0     # in frames: summing 0.08 s floats drifts below the limits
        self._speech = False

    def push(self, rms: float) -> str | None:
        """Returns None while the command goes on, else why it ended: 'end', 'no_speech' or 'max'."""
        return self.step(rms > max(self.noise * self.ratio, self.min_rms))

    def step(self, speech: bool) -> str | None:
        """Like `push`, for a frame already judged speech or not."""
        self._elapsed += 1
        if speech:
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


class SileroVAD:
    """Silero VAD v6 (shipped with faster-whisper) run as a stream: 512-sample chunks with 64 samples of
    context and the recurrent state carried over, a speech probability per 32 ms.

    Loudness cannot tell a voice from a video playing next to the microphone; this model judges speech
    itself, so a command is not cut off mid-word, nor taken for silence, because a video was loud."""
    CHUNK, CONTEXT = 512, 64

    def __init__(self, threshold: float = 0.5):
        from faster_whisper.vad import get_vad_model
        self._session = get_vad_model().session
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._context = np.zeros(self.CONTEXT, np.float32)
        self._pending = np.zeros(0, np.float32)

    def push(self, frame: np.ndarray) -> list[float]:
        """Speech probabilities of the 32 ms chunks this frame completes."""
        x = frame.astype(np.float32) / 32768.0 if frame.dtype == np.int16 else frame.astype(np.float32)
        self._pending = np.concatenate([self._pending, x])
        probs = []
        while len(self._pending) >= self.CHUNK:
            chunk, self._pending = self._pending[:self.CHUNK], self._pending[self.CHUNK:]
            batch = np.concatenate([self._context, chunk])[None, :]
            out, self._h, self._c = self._session.run(None, {"input": batch, "h": self._h, "c": self._c})
            probs.append(float(np.asarray(out).reshape(-1)[-1]))
            self._context = chunk[-self.CONTEXT:]
        return probs

    def speech(self, frame: np.ndarray) -> bool:
        return max(self.push(frame), default=0.0) >= self.threshold


class Ducker(Protocol):
    def duck(self) -> None: ...
    def restore(self) -> None: ...


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
                 input_device: int | str | None = None, vad: SileroVAD | None = None,
                 ducker: Ducker | None = None):
        self.bus, self.wake, self.stt = bus, wake, stt
        self.endpointer = endpointer or Endpointer()
        self.beep = beep or _beep
        self.input_device = input_device
        self.vad, self.ducker = vad, ducker

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
        try:
            self._loop(frames, stop)
        finally:
            if self.ducker:
                self.ducker.restore()               # never leave the machine quiet

    def _loop(self, frames: queue.Queue, stop: threading.Event) -> None:
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
                    if self.ducker:
                        self.ducker.duck()
                    if self.vad:
                        self.vad.reset()
                    self.endpointer.start()
                    recording = []
                continue
            recording.append(frame)
            ended = self.endpointer.step(self.vad.speech(frame)) if self.vad else self.endpointer.push(level)
            if ended is None:
                continue
            if self.ducker:
                self.ducker.restore()               # before the command runs: «громче» must act on the real level
            if ended == "no_speech":
                self.bus.publish(Event("voice.timeout", source="voice_listener"))
                print("(no command heard)", flush=True)
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
