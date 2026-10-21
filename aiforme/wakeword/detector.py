"""The trained wake-word model at run time: ONNX Runtime on a sliding 1.2 s window, no PyTorch.

Every 80 ms microphone frame moves the window; the detector fires when the probability stays above the
threshold for `consecutive` windows in a row, then ignores the next `refractory_s` seconds so one word does
not trigger twice. The same class scores the test recordings in train.py, so the evaluation measures exactly
what the assistant runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aiforme.wakeword.features import SAMPLE_RATE, WINDOW, log_mel

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL = MODELS_DIR / "wakeword.onnx"
CONFIG = MODELS_DIR / "wakeword.json"
HOP = 1280                      # 80 ms, the listener's frame


class Detector:
    def __init__(self, model: Path = MODEL, config: Path = CONFIG, threshold: float | None = None):
        import onnxruntime as ort
        if not Path(model).exists():
            raise FileNotFoundError(f"{model} not found: train it with python -m aiforme.wakeword.train")
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        self.threshold = cfg["threshold"] if threshold is None else threshold
        self.consecutive = cfg.get("consecutive", 2)
        self.refractory = int(cfg.get("refractory_s", 1.5) * SAMPLE_RATE / HOP)
        self._session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        self._buffer = np.zeros(WINDOW, np.float32)
        self._above = 0
        self._cooldown = 0

    def probability(self, window: np.ndarray) -> float:
        feats = log_mel(window)[None, None]
        logit = self._session.run(None, {self._input: feats})[0].reshape(-1)[0]
        return float(1.0 / (1.0 + np.exp(-logit)))

    def detected(self, frame: np.ndarray) -> bool:
        """Feeds one frame (int16 or float); True when the wake word has just been said."""
        x = frame.astype(np.float32) / 32768.0 if frame.dtype == np.int16 else frame.astype(np.float32)
        self._buffer = np.concatenate([self._buffer[len(x):], x])
        if self._cooldown:
            self._cooldown -= 1
            return False
        self._above = self._above + 1 if self.probability(self._buffer) >= self.threshold else 0
        if self._above >= self.consecutive:
            self._above, self._cooldown = 0, self.refractory
            return True
        return False

    def count(self, audio: np.ndarray) -> int:
        """How many times the wake word fires in a recording, streamed frame by frame from a fresh state."""
        self.reset()
        hits = sum(self.detected(audio[i:i + HOP]) for i in range(0, len(audio) - HOP + 1, HOP))
        self.reset()
        return hits
