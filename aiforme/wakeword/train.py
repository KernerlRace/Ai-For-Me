"""Trains the wake-word CNN from scratch, picks its threshold, exports ONNX and scores the test session.

    python -m aiforme.wakeword.train                  # Windows voices + your recordings
    python -m aiforme.wakeword.train --no-recorded    # Windows voices only, for comparison

Training data: the synthetic clips and the `train` recordings, split by file into fit (85%) and validation
(15%). The threshold is the lowest one at which no validation negative window fires, but at least 0.5.
The `test` recordings — another session — are never touched until the end, and are scored by streaming
them through the exported model exactly as the assistant does: recall over the test takes of the wake word,
false accepts per hour over the test room recording.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from aiforme.wakeword.dataset import build, files, read_wav
from aiforme.wakeword.detector import CONFIG, MODEL, Detector
from aiforme.wakeword.features import FRAMES, N_MELS, SAMPLE_RATE

RESULTS = MODEL.parents[1] / "eval" / "results"


def make_model(channels=(16, 32, 64, 64)):
    import torch.nn as nn

    layers, c_in = [], 1
    for i, c in enumerate(channels):
        layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU()]
        if i < len(channels) - 1:
            layers.append(nn.MaxPool2d(2))
        c_in = c
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c_in, 1))


class Normalized:
    """Per-band mean and std folded into the exported graph, so the detector feeds raw log-mels."""

    def __new__(cls, model, mean, std):
        import torch
        import torch.nn as nn

        class _N(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.tensor(mean).view(1, 1, -1, 1))
                self.register_buffer("std", torch.tensor(std).view(1, 1, -1, 1))
                self.model = model

            def forward(self, x):
                return self.model((x - self.mean) / self.std)

        return _N()


def load(kind: str, split: str, synth: bool, recorded: bool) -> list[np.ndarray]:
    paths = (files(kind, split, "synth") if synth else []) + (files(kind, split) if recorded else [])
    return [read_wav(p) for p in paths]


def split_files(clips: list, frac: float, rng: np.random.Generator) -> tuple[list, list]:
    idx = rng.permutation(len(clips))
    n_val = int(round(len(clips) * frac))
    return [clips[i] for i in idx[n_val:]], [clips[i] for i in idx[:n_val]]


def coloured_noise(rng: np.random.Generator, seconds: float = 30.0) -> list[np.ndarray]:
    """White and pink noise, so there is something to mix in even before any room is recorded."""
    n = int(seconds * SAMPLE_RATE)
    white = rng.standard_normal(n).astype(np.float32)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    pink = np.fft.irfft(spectrum / np.sqrt(np.arange(1, len(spectrum) + 1)), n).astype(np.float32)
    return [white / np.abs(white).max(), pink / np.abs(pink).max()]


def train(recorded: bool = True, epochs: int = 25, seed: int = 0) -> dict:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    clips_fit, clips_val, recorded_val = {}, {}, {}
    for kind in ("positive", "negative", "ambient"):
        # synthetic and recorded clips are split separately, so validation has its share of each
        fit_s, val_s = split_files(load(kind, "train", synth=kind != "ambient", recorded=False), 0.15, rng)
        # recordings are always split the same way, so a synthetic-only model is scored on the same takes
        fit_r, val_r = split_files(load(kind, "train", synth=False, recorded=True), 0.15, rng)
        if not recorded:
            fit_r, val_used = [], []
        else:
            val_used = val_r
        clips_fit[kind], clips_val[kind], recorded_val[kind] = fit_s + fit_r, val_s + val_used, val_r
    noise = clips_fit["ambient"] + coloured_noise(rng)
    t0 = time.perf_counter()
    X, y = build(clips_fit, noise, seed=seed)
    Xv, yv = build(clips_val, noise, seed=seed + 1, ambient_windows=800)
    print(f"windows: fit {len(y)} ({int(y.sum())} positive), val {len(yv)} ({int(yv.sum())} positive), "
          f"{time.perf_counter() - t0:.0f} s", flush=True)

    mean, std = X.mean(axis=(0, 2)), X.std(axis=(0, 2)) + 1e-5
    norm = lambda a: (a - mean[None, :, None]) / std[None, :, None]  # noqa: E731
    Xt, yt = torch.tensor(norm(X))[:, None], torch.tensor(y)
    Xvt = torch.tensor(norm(Xv))[:, None]

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(yt) / 256))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - y.sum()) / max(y.sum(), 1)))
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        total = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            xb = _spec_augment(Xt[b].clone())
            loss = loss_fn(model(xb).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        pv = _predict(model, Xvt)
        print(f"epoch {epoch + 1:2d}  loss {total / len(yt):.4f}  val recall@0.5 {np.mean(pv[yv == 1] >= .5):.3f}"
              f"  val false@0.5 {np.mean(pv[yv == 0] >= .5):.4f}", flush=True)

    pv = _predict(model, Xvt)
    threshold = float(max(0.5, min(0.99, np.max(pv[yv == 0]) + 0.01)))
    model_path = MODEL if recorded else MODEL.with_name("wakeword_synth_only.onnx")
    config_path = CONFIG if recorded else CONFIG.with_name("wakeword_synth_only.json")
    MODEL.parent.mkdir(exist_ok=True)
    wrapped = Normalized(model.eval(), mean.astype(np.float32), std.astype(np.float32))
    torch.onnx.export(wrapped, torch.zeros(1, 1, N_MELS, FRAMES), str(model_path), input_names=["features"],
                      output_names=["logit"], dynamic_axes={"features": {0: "batch"}}, dynamo=False)
    config = {"threshold": threshold, "consecutive": 2, "refractory_s": 1.5, "trained_on_recordings": recorded,
              "parameters": sum(p.numel() for p in model.parameters()), "epochs": epochs}
    config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")

    report = {**config, "fit_windows": len(y), "val_windows": len(yv),
              "val_recall_at_threshold": float(np.mean(pv[yv == 1] >= threshold)),
              "val_streaming": streaming(Detector(model_path, config_path), recorded_val["positive"],
                                         recorded_val["ambient"]),
              "test": evaluate_test(model_path, config_path)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = "wakeword" if recorded else "wakeword_synth_only"
    (RESULTS / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def evaluate_test(model_path=MODEL, config_path=CONFIG) -> dict | None:
    return streaming(Detector(model_path, config_path), [read_wav(p) for p in files("positive", "test")],
                     [read_wav(p) for p in files("ambient", "test")])


def streaming(det: Detector, positives: list[np.ndarray], ambient: list[np.ndarray]) -> dict | None:
    """Recordings streamed through the detector as the assistant hears them: recall over takes of the wake
    word (each padded with 0.6 s of silence), false accepts over the room recording."""
    if not positives and not ambient:
        return None
    out = {}
    if positives:
        pad = np.zeros(int(0.6 * SAMPLE_RATE), np.float32)
        hits = [det.count(np.concatenate([pad, clip, pad])) > 0 for clip in positives]
        out["recall"] = float(np.mean(hits))
        out["positives"] = len(positives)
    if ambient:
        audio = np.concatenate(ambient)
        hours = len(audio) / SAMPLE_RATE / 3600
        out["false_accepts"] = det.count(audio)
        out["ambient_minutes"] = round(hours * 60, 1)
        out["false_accepts_per_hour"] = round(out["false_accepts"] / hours, 2)
    return out


def _spec_augment(x):
    """Masks up to 6 mel bands and up to 15 frames per example."""
    import torch
    n, _, mels, frames = x.shape
    f = torch.randint(0, 7, (n,))
    f0 = (torch.rand(n) * (mels - f)).long()
    t = torch.randint(0, 16, (n,))
    t0 = (torch.rand(n) * (frames - t)).long()
    bands = torch.arange(mels)[None, :]
    steps = torch.arange(frames)[None, :]
    fmask = (bands >= f0[:, None]) & (bands < (f0 + f)[:, None])
    tmask = (steps >= t0[:, None]) & (steps < (t0 + t)[:, None])
    x[fmask[:, None, :, None].expand_as(x)] = 0.0
    x[tmask[:, None, None, :].expand_as(x)] = 0.0
    return x


def _predict(model, X) -> np.ndarray:
    import torch
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(torch.cat([model(X[i:i + 1024]) for i in range(0, len(X), 1024)])).squeeze(1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-recorded", action="store_true", help="train on the Windows voices only")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    train(recorded=not args.no_recorded, epochs=args.epochs)


if __name__ == "__main__":
    main()
