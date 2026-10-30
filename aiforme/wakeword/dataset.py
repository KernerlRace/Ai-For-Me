"""From recordings to training windows: where the word is, which 1.5 s windows hold it, and augmentation.

A positive window holds the whole wake word somewhere inside it. Windows that end halfway through the
word ("Ама...") are negatives on purpose: without them the detector fires before the word is finished,
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


FRAME = 160                     # 10 ms


def _segments(audio: np.ndarray, max_gap_s: float) -> list[tuple[int, int]]:
    """Loud stretches of a clip, as sample ranges. Loud means within 30 dB of the loudest 10 ms and at least
    10 dB above the background. The threshold hangs from the peak, not from the noise: a microphone with
    noise suppression has a near-digital-silence floor, from which breath, a key click or the beep's echo
    would all look like speech. Gaps shorter than `max_gap_s` are bridged; bursts under 100 ms are clicks."""
    n = len(audio) // FRAME
    if n < 3:
        return []
    level = 10 * np.log10((audio[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1) + 1e-10)
    floor, peak = np.percentile(level, 10), level.max()
    if peak - floor < 12:
        return []
    active = level > max(peak - 30, floor + 10)
    runs, start, gap = [], None, 0
    for i, on in enumerate(active):
        if on:
            if start is None:
                start = i
            gap, last = 0, i
        elif start is not None:
            gap += 1
            if gap * FRAME / SAMPLE_RATE >= max_gap_s:
                runs.append((start, last + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, last + 1))
    return [(a * FRAME, b * FRAME) for a, b in runs if (b - a) * FRAME >= 0.1 * SAMPLE_RATE]


def speaker_of(path: Path) -> str:
    """'anna_20261022-..._003.wav' -> 'anna'; the first recordings had no prefix and are mine."""
    first = path.stem.split("_")[0]
    return "me" if first[:1].isdigit() else first


def speech_span(audio: np.ndarray) -> tuple[int, int] | None:
    """From the start of the first loud stretch to the end of the last; None if nothing stands out."""
    segs = _segments(audio, max_gap_s=0.1)
    return (segs[0][0], segs[-1][1]) if segs else None


def word_span(audio: np.ndarray) -> tuple[int, int] | None:
    """Where the wake word is: the first stretch of speech followed by a pause of at least 200 ms — the
    whole speech when it is said alone, the part before the comma when a command follows. Without such a
    pause (the command said in one breath) the word is taken as the first 1.0 s."""
    segs = _segments(audio, max_gap_s=0.2)
    if not segs:
        return None
    start, end = segs[0]
    if end - start > int(1.45 * SAMPLE_RATE):    # longer than a slow «Амадеус»: the command ran on
        end = start + int(1.0 * SAMPLE_RATE)
    return start, end


# --- augmentation ---------------------------------------------------------------------------------------

@dataclass
class Augmenter:
    noise: list[np.ndarray]           # background recordings to mix in
    rng: np.random.Generator

    whisper: float = 0.0              # share of clips turned into a whisper first

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r = self.rng
        if self.whisper and r.random() < self.whisper:
            x = whisperize(x, r)
        if r.random() < 0.5:                                  # tempo and pitch together, ±10%
            x = _resample(x, r.uniform(0.9, 1.1))
        if r.random() < 0.3:
            x = np.convolve(x, _room(r), mode="full")[: len(x)]
        x = x * 10 ** (r.uniform(-12, 6) / 20)
        if self.noise and r.random() < 0.7:
            x = _mix(x, self.noise[r.integers(len(self.noise))], r.uniform(3, 30), r)
        return np.clip(x, -1.0, 1.0)


def whisperize(x: np.ndarray, r: np.random.Generator, n_fft: int = 512, hop: int = 128,
               lifter: int = 24) -> np.ndarray:
    """The same words, whispered: a whisper is speech without the voice — the vocal tract still shapes the
    sound, but noise excites it instead of the vocal folds. Each frame keeps its spectral envelope (the log
    spectrum smoothed by keeping its first `lifter` cepstral coefficients, which drops the harmonics of the
    pitch) and gets a random phase, so the formants stay where they were and the pitch is gone."""
    if len(x) < n_fft:
        return x
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    spec = np.fft.rfft(x[idx] * win, axis=1)
    cep = np.fft.irfft(np.log(np.abs(spec) + 1e-6), axis=1)
    cep[:, lifter:n_fft - lifter + 1] = 0.0
    envelope = np.exp(np.fft.rfft(cep, axis=1).real)
    # and the low end thins out: without the voice there is little energy below a kilohertz
    f = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
    envelope = envelope * (f / (f + r.uniform(500, 1200)))
    phase = np.exp(2j * np.pi * r.random(envelope.shape))
    frames = np.fft.irfft(envelope * phase, n_fft, axis=1) * win
    # overlap-add (np.add.at with a broadcast window returned garbage and crashed under numpy 2.5)
    out = np.bincount(idx.ravel(), weights=frames.ravel(), minlength=len(x))
    norm = np.bincount(idx.ravel(), weights=np.broadcast_to(win ** 2, idx.shape).ravel(), minlength=len(x))
    out = out / np.maximum(norm, 1e-3)
    rms_in, rms_out = np.sqrt((x ** 2).mean()) + 1e-9, np.sqrt((out ** 2).mean()) + 1e-9
    return (out * rms_in / rms_out).astype(np.float32)


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
          ambient_windows: int = 4000, whisper: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """clips: 'positive', 'negative', 'ambient' -> waveforms. Returns features [N, mels, frames] and labels."""
    r = np.random.default_rng(seed)
    aug = Augmenter(noise, r, whisper)
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
