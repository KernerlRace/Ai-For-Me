"""Trains the wake-word CNN from scratch, picks its threshold, exports ONNX and scores the test session.

    python -m aiforme.wakeword.train                  # Windows voices + your recordings
    python -m aiforme.wakeword.train --no-recorded    # Windows voices only, for comparison

Training data: the synthetic clips and the `train` recordings, split by file into fit (85%) and validation
(15%). Two rounds: the first model is run over the training room audio and the other phrases, the windows it
mistakes for the wake word are added as hard negatives, and a second model is trained from scratch on everything.
The threshold is the lowest at which the held-out real audio — room recordings and my other phrases, streamed as
the detector hears them — never fires; how often the synthetic phrases fire at it is reported alongside. Trains on
the GPU if any. The `test` recordings — another session — are never touched until the end, and are scored by
streaming them through the exported model exactly as the assistant does: recall over the test takes of the wake
word, false accepts per hour over the test room recording.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from aiforme.wakeword.dataset import Augmenter, build, files, read_wav, speaker_of
from aiforme.wakeword.dataset import _fit as _fit_len
from aiforme.wakeword.detector import CONFIG, HOP, MODEL, Detector
from aiforme.wakeword.features import FRAMES, N_MELS, SAMPLE_RATE, WINDOW, log_mel

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


def train(recorded: bool = True, epochs: int = 25, seed: int = 0, mine: bool = True) -> dict:
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    clips_fit, synth_val, recorded_val = {}, {}, {}
    for kind in ("positive", "negative", "ambient"):
        # synthetic and recorded clips are split separately, so validation has its share of each
        fit_s, val_s = split_files(load(kind, "train", synth=kind != "ambient", recorded=False), 0.15, rng)
        # recordings are always split the same way, so a synthetic-only model is scored on the same takes
        fit_r, val_r = split_files(load(kind, "train", synth=False, recorded=True), 0.15, rng)
        if not recorded:
            fit_r = []
        clips_fit[kind], synth_val[kind], recorded_val[kind] = fit_s + fit_r, val_s, val_r
    noise = clips_fit["ambient"] + coloured_noise(rng)
    ambient_s = sum(len(a) for a in clips_fit["ambient"]) / SAMPLE_RATE
    t0 = time.perf_counter()
    X, y = build(clips_fit, noise, seed=seed, ambient_windows=int(min(12000, max(2000, ambient_s * 3))))
    print(f"windows: {len(y)} ({int(y.sum())} positive), {time.perf_counter() - t0:.0f} s", flush=True)

    rounds = []
    model, mean, std = _fit(X, y, epochs, seed)
    rounds.append(_validate(model, mean, std, synth_val, recorded_val, "round 1"))
    if mine and clips_fit["ambient"]:
        # hard negatives: the windows of the training room audio and of the other phrases that the first model
        # mistakes for the word. The phrases matter for a new voice: three random windows per phrase are few next to
        # fifteen per wake word, and v3 fired on 23 of 60 plain commands said by a second speaker. Each source has
        # its own cap: in one pool a voice the model has not learned yet fills it, and the room goes unmined
        aug = Augmenter(noise, rng)
        hard = np.concatenate([_mine(model, mean, std, clips_fit["ambient"], aug),
                               _mine(model, mean, std, clips_fit["negative"], aug)])
        print(f"hard negatives: {len(hard)} windows", flush=True)
        X = np.concatenate([X, hard])
        y = np.concatenate([y, np.zeros(len(hard), np.float32)])
        model, mean, std = _fit(X, y, epochs, seed)                # from scratch, with the hard negatives
        rounds.append(_validate(model, mean, std, synth_val, recorded_val, "round 2"))
        rounds[-1]["hard_negatives"] = len(hard)
    final = rounds[-1]

    model_path = MODEL if recorded else MODEL.with_name("wakeword_synth_only.onnx")
    config_path = CONFIG if recorded else CONFIG.with_name("wakeword_synth_only.json")
    MODEL.parent.mkdir(exist_ok=True)
    wrapped = Normalized(model.cpu().eval(), mean.astype(np.float32), std.astype(np.float32))
    torch.onnx.export(wrapped, torch.zeros(1, 1, N_MELS, FRAMES), str(model_path), input_names=["features"],
                      output_names=["logit"], dynamic_axes={"features": {0: "batch"}}, dynamo=False)
    config = {"threshold": final["threshold"], "consecutive": CONSECUTIVE, "refractory_s": REFRACTORY_S,
              "trained_on_recordings": recorded, "parameters": sum(p.numel() for p in model.parameters()),
              "epochs": epochs, "fit_windows": len(y), "fit_ambient_minutes": round(ambient_s / 60, 1)}
    config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")

    report = {**config, "rounds": rounds,
              "val_streaming": streaming(Detector(model_path, config_path), recorded_val["positive"],
                                         recorded_val["ambient"]),
              "test": evaluate_test(model_path, config_path)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = "wakeword" if recorded else "wakeword_synth_only"
    (RESULTS / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


CONSECUTIVE = 2
REFRACTORY_S = 1.5
THRESHOLDS = np.unique(np.round(np.concatenate([np.arange(0.5, 0.99, 0.01), np.arange(0.99, 0.9995, 0.001)]), 4))


def _fit(X: np.ndarray, y: np.ndarray, epochs: int, seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean, std = X.mean(axis=(0, 2)), X.std(axis=(0, 2)) + 1e-5
    Xt = torch.tensor((X - mean[None, :, None]) / std[None, :, None])[:, None].to(device)
    yt = torch.tensor(y).to(device)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(yt) / 256))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), device=device))
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), device=device)
        total = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = loss_fn(model(_spec_augment(Xt[b].clone())).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:2d}  loss {total / len(yt):.4f}", flush=True)
    return model.eval(), mean, std


def stream_probs(model, mean, std, audio: np.ndarray) -> np.ndarray:
    """The probability the detector computes after each 80 ms frame of `audio`, from a zero-filled buffer.
    The log-mel of the whole recording is computed once and sliced: a window starts every 8 frames of
    10 ms, so each slice equals the features of that window."""
    import torch
    padded = np.concatenate([np.zeros(WINDOW - HOP, np.float32), audio.astype(np.float32)])
    feats = (log_mel(padded) - mean[:, None]) / std[:, None]
    step = HOP // 160
    n = 1 + (feats.shape[1] - FRAMES) // step
    out = []
    with torch.no_grad():
        for i in range(0, n, 2048):
            idx = range(i, min(n, i + 2048))
            batch = np.stack([feats[:, k * step:k * step + FRAMES] for k in idx])[:, None]
            x = torch.tensor(batch.astype(np.float32)).to(_device(model))
            out.append(torch.sigmoid(model(x)).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _device(model):
    params = list(model.parameters())
    return params[0].device if params else "cpu"


def count_hits(probs: np.ndarray, threshold: float, refractory_s: float = REFRACTORY_S) -> int:
    """The detector's decision rule (detector.py) applied to a precomputed probability sequence."""
    refractory = int(refractory_s * SAMPLE_RATE / HOP)
    above = cooldown = hits = 0
    for p in probs:
        if cooldown:
            cooldown -= 1
            continue
        above = above + 1 if p >= threshold else 0
        if above >= CONSECUTIVE:
            hits, above, cooldown = hits + 1, 0, refractory
    return hits


def _validate(model, mean, std, synth_val, recorded_val, name: str) -> dict:
    """Streams the held-out recordings and picks the lowest threshold at which none of the real negatives —
    the room audio and my other phrases — fires. The Windows voices are not used for it: their «Амадей» and
    "Amadeo" score higher than anything the microphone hears, and obeying them cost recall on real speech
    for no fewer false accepts in the room. How often they fire is reported instead."""
    pad = np.zeros(int(0.6 * SAMPLE_RATE), np.float32)

    def clips(cs):
        return [stream_probs(model, mean, std, np.concatenate([pad, c, pad])) for c in cs]

    room = stream_probs(model, mean, std, np.concatenate(recorded_val["ambient"])) if recorded_val["ambient"] else None
    real = ([room] if room is not None else []) + clips(recorded_val["negative"])
    synthetic = clips(synth_val["negative"])
    positives = clips(recorded_val["positive"])
    threshold = float(THRESHOLDS[-1])
    for t in THRESHOLDS:
        if sum(count_hits(p, t) for p in real) == 0:
            threshold = float(t)
            break
    out = {
        "threshold": threshold,
        "val_recall": float(np.mean([count_hits(p, threshold) > 0 for p in positives])) if positives else None,
        "val_room_minutes": round(sum(len(a) for a in recorded_val["ambient"]) / SAMPLE_RATE / 60, 1),
        "val_recorded_phrases_fired": f"{sum(count_hits(p, threshold) > 0 for p in clips(recorded_val['negative']))}"
                                      f" of {len(recorded_val['negative'])}",
        "val_synthetic_phrases_fired": f"{sum(count_hits(p, threshold) > 0 for p in synthetic)} of {len(synthetic)}",
    }
    print(f"{name}: {out}", flush=True)
    return out


def _mine(model, mean, std, clips: list[np.ndarray], aug, min_p: float = 0.1, cap: int = 4000) -> np.ndarray:
    """Features of the highest-scoring windows of negative audio, each once as it is and once augmented."""
    found = []
    for clip in clips:
        probs = stream_probs(model, mean, std, clip)
        padded = np.concatenate([np.zeros(WINDOW - HOP, np.float32), clip])
        found += [(p, padded, k) for k, p in enumerate(probs) if p >= min_p]
    found.sort(key=lambda f: -f[0])
    out = []
    for _, padded, k in found[: cap // 2]:
        w = padded[k * HOP:k * HOP + WINDOW]
        out += [log_mel(w), log_mel(_fit_len(aug(w)))]
    return np.stack(out) if out else np.zeros((0, N_MELS, FRAMES), np.float32)


def evaluate_test(model_path=MODEL, config_path=CONFIG) -> dict | None:
    det = Detector(model_path, config_path)
    takes = files("positive", "test")
    out = streaming(det, [read_wav(p) for p in takes], [read_wav(p) for p in files("ambient", "test")])
    speakers = sorted({speaker_of(p) for p in takes})
    if out and len(speakers) > 1:
        out["recall_by_speaker"] = {s: streaming(det, [read_wav(p) for p in takes if speaker_of(p) == s], [])
                                    ["recall"] for s in speakers}
    return out


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
    d = x.device
    f = torch.randint(0, 7, (n,), device=d)
    f0 = (torch.rand(n, device=d) * (mels - f)).long()
    t = torch.randint(0, 16, (n,), device=d)
    t0 = (torch.rand(n, device=d) * (frames - t)).long()
    bands = torch.arange(mels, device=d)[None, :]
    steps = torch.arange(frames, device=d)[None, :]
    fmask = (bands >= f0[:, None]) & (bands < (f0 + f)[:, None])
    tmask = (steps >= t0[:, None]) & (steps < (t0 + t)[:, None])
    x[fmask[:, None, :, None].expand_as(x)] = 0.0
    x[tmask[:, None, None, :].expand_as(x)] = 0.0
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-recorded", action="store_true", help="train on the Windows voices only")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--no-mining", action="store_true", help="one round, without hard negatives")
    parser.add_argument("--eval-only", action="store_true", help="score the current model on the test session")
    args = parser.parse_args()
    if args.eval_only:
        print(json.dumps(evaluate_test(), indent=1))
    else:
        train(recorded=not args.no_recorded, epochs=args.epochs, mine=not args.no_mining)


if __name__ == "__main__":
    main()
