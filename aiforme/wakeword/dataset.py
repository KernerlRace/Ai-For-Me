"""From recordings to training windows: where the word is, which 1.2 s windows hold it, and augmentation.

A positive window holds the whole wake word somewhere inside it. Windows that end halfway through the
word ("Джар...") are negatives on purpose: without them the detector fires before the word is finished,
and on every word that merely starts the same way. Negative windows are cut at random from the other
phrases and from the room recordings, which also serve as background noise for mixing.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aiforme.core.config import data_dir
from aiforme.wakeword.features import SAMPLE_RATE, WINDOW, log_mel

ROOT_NAME = "wakeword"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def files(kind: str, split: str, source: str = "recorded") -> list[Path]:
    """source: 'recorded' (your microphone) or 'synth' (Windows voices; train only)."""
    root = data_dir() / ROOT_NAME
    folder = root / "synth" / kind if source == "synth" else root / kind / split
    return sorted(folder.glob("*.wav")) if folder.exists() else []


def speech_span(audio: np.ndarray, frame: int = 160) -> tuple[int, int] | None:
    """First and last sample of the louder part of a clip: frames above 25% of the way from the noise
    level (10th percentile) to the loudest frame, in dB. None if nothing stands out."""
    n = len(audio) // frame
    if n < 3:
        return None
    level = 10 * np.log10((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-10)
    floor, peak = np.percentile(level, 10), level.max()
    if peak - floor < 12:
        return None
    active = np.flatnonzero(level > floor + 0.25 * (peak - floor))
    return int(active[0] * frame), int((active[-1] + 1) * frame)


def word_span(audio: np.ndarray) -> tuple[int, int] | None:
    """Where the wake word is. Alone, it is the whole speech; before a command it is the first burst of
    speech, which ends at the first pause of at least 120 ms or, failing that, 0.9 s after the start."""
    span = speech_span(audio)
    if span is None:
        return None
    start, end = span
    if end - start <= int(1.0 * SAMPLE_RATE):
        return span
    frame = 160
    seg = audio[start:end]
    n = len(seg) // frame
    level = 10 * np.log10((seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-10)
    quiet = level < np.percentile(level, 90) - 20
    run = 0
    for i in range(int(0.25 * SAMPLE_RATE / frame), min(n, int(1.2 * SAMPLE_RATE / frame))):
        run = run + 1 if quiet[i] else 0
        if run >= 12:
            return start, start + (i - run + 1) * frame
    return start, start + int(0.9 * SAMPLE_RATE)


# --- augmentation ---------------------------------------------------------------------------------------

@dataclass
class Augmenter:
    noise: list[np.ndarray]           # background recordings to mix in
    rng: np.random.Generator

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r = self.rng
        if r.random() < 0.5:                                  # tempo and pitch together, ±10%
            x = _resample(x, r.uniform(0.9, 1.1))
        if r.random() < 0.3:
            x = np.convolve(x, _room(r), mode="full")[: len(x)]
        x = x * 10 ** (r.uniform(-12, 6) / 20)
        if self.noise and r.random() < 0.7:
            x = _mix(x, self.noise[r.integers(len(self.noise))], r.uniform(3, 30), r)
        return np.clip(x, -1.0, 1.0)


def _resample(x: np.ndarray, factor: float) -> np.ndarray:
    n = int(len(x) / factor)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _room(r: np.random.Generator) -> np.ndarray:
    """A crude room: exponentially decaying noise, 0.1–0.5 s."""
    t60 = r.uniform(0.1, 0.5)
    n = int(t60 * SAMPLE_RATE)
    ir = r.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    return (ir / np.abs(ir).sum() * 4).astype(np.float32)


def _mix(x: np.ndarray, noise: np.ndarray, snr_db: float, r: np.random.Generator) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, len(x) // len(noise) + 1)
    start = r.integers(0, len(noise) - len(x) + 1)
    n = noise[start:start + len(x)]
    p_x, p_n = (x ** 2).mean() + 1e-10, (n ** 2).mean() + 1e-10
    return x + n * np.sqrt(p_x / (p_n * 10 ** (snr_db / 10)))


# --- windows --------------------------------------------------------------------------------------------

def _window(x: np.ndarray, start: int) -> np.ndarray:
    """WINDOW samples from `start`, padded with near-silence where the clip runs out."""
    out = np.zeros(WINDOW, np.float32)
    lo, hi = max(0, start), min(len(x), start + WINDOW)
    if hi > lo:
        out[lo - start:hi - start] = x[lo:hi]
    return out


def _fit(x: np.ndarray) -> np.ndarray:
    """Back to WINDOW samples after a tempo change: trimmed evenly from both ends, or padded at the end."""
    if len(x) >= WINDOW:
        start = (len(x) - WINDOW) // 2
        return x[start:start + WINDOW]
    return np.pad(x, (0, WINDOW - len(x)))


def positive_windows(x: np.ndarray, k: int, r: np.random.Generator) -> list[np.ndarray]:
    span = word_span(x)
    if span is None:
        return []
    s, e = span
    pad = int(0.05 * SAMPLE_RATE)
    lo, hi = e + pad - WINDOW, s - pad          # window starts that keep the whole word inside
    if lo > hi:                                 # the word does not fit: a cut-off window would teach noise
        return []
    return [_window(x, int(r.integers(lo, hi + 1))) for _ in range(k)]


def partial_windows(x: np.ndarray, k: int, r: np.random.Generator) -> list[np.ndarray]:
    """Windows ending in the first 40% of the word."""
    span = word_span(x)
    if span is None:
        return []
    s, e = span
    cut_lo, cut_hi = s + int(0.08 * SAMPLE_RATE), s + int(0.4 * (e - s))
    if cut_hi <= cut_lo:
        return []
    return [_window(x, int(r.integers(cut_lo, cut_hi)) - WINDOW) for _ in range(k)]


def random_windows(x: np.ndarray, k: int, r: np.random.Generator) -> list[np.ndarray]:
    hi = max(0, len(x) - WINDOW)
    return [_window(x, int(r.integers(0, hi + 1))) for _ in range(k)]


def build(clips: dict[str, list[np.ndarray]], noise: list[np.ndarray], seed: int = 0,
          per_positive: int = 15, per_negative: int = 3, per_partial: int = 2,
          ambient_windows: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """clips: 'positive', 'negative', 'ambient' -> waveforms. Returns features [N, mels, frames] and labels."""
    r = np.random.default_rng(seed)
    aug = Augmenter(noise, r)
    xs, ys = [], []

    def add(windows, label, augment=True):
        for w in windows:
            xs.append(log_mel(_fit(aug(w)) if augment else w))
            ys.append(label)

    for clip in clips.get("positive", []):
        add(positive_windows(clip, per_positive, r), 1)
        add(partial_windows(clip, per_partial, r), 0)
    for clip in clips.get("negative", []):
        add(random_windows(clip, per_negative, r), 0)
    ambient = clips.get("ambient", [])
    for i in range(ambient_windows if ambient else 0):
        add(random_windows(ambient[i % len(ambient)], 1, r), 0, augment=r.random() < 0.3)
    return np.stack(xs), np.array(ys, dtype=np.float32)
