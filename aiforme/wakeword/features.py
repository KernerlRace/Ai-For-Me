"""Log-mel features in NumPy, shared by training and the live detector, so the assistant needs no PyTorch.

25 ms Hann windows every 10 ms, 40 mel bands up to 8 kHz, log of the power with a floor. A detector window
of 1.5 s is 148 frames; the model sees a 40 × 148 image.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 512
WIN = 400               # 25 ms
HOP = 160               # 10 ms
N_MELS = 40
WINDOW_S = 1.5         # «Амадеус» said slowly takes up to 1.4 s
WINDOW = int(WINDOW_S * SAMPLE_RATE)
FRAMES = 1 + (WINDOW - WIN) // HOP


def _mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SAMPLE_RATE,
                   fmin: float = 20.0, fmax: float = 8000.0) -> np.ndarray:
    """Triangular filters, [n_mels, n_fft // 2 + 1]."""
    edges = _hz(np.linspace(_mel(np.array(fmin)), _mel(np.array(fmax)), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (bins - lo) / (mid - lo)
        down = (hi - bins) / (hi - mid)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


_FB = mel_filterbank()
_HANN = np.hanning(WIN).astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """int16 or float samples at 16 kHz -> [N_MELS, frames] float32."""
    x = audio.astype(np.float32)
    if audio.dtype == np.int16:
        x /= 32768.0
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * _HANN
    power = np.abs(np.fft.rfft(frames, N_FFT)) ** 2
    return np.log(power @ _FB.T + 1e-6).T.astype(np.float32)
