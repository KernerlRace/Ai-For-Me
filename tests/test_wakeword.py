import numpy as np
import pytest

from aiforme.wakeword.dataset import (
    Augmenter,
    _fit,
    partial_windows,
    positive_windows,
    random_windows,
    speech_span,
    word_span,
)
from aiforme.wakeword.features import FRAMES, N_MELS, SAMPLE_RATE, WINDOW, log_mel, mel_filterbank

SR = SAMPLE_RATE


def tone(seconds, freq=300.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def clip(*parts):
    """('silence', 0.3), ('speech', 0.5) ... -> a waveform with faint noise under everything."""
    rng = np.random.default_rng(0)
    audio = np.concatenate([tone(s) if kind == "speech" else np.zeros(int(s * SR), np.float32)
                            for kind, s in parts])
    return audio + 0.001 * rng.standard_normal(len(audio)).astype(np.float32)


def test_features_shape_and_filterbank():
    assert log_mel(np.zeros(WINDOW, np.int16)).shape == (N_MELS, FRAMES)
    fb = mel_filterbank()
    assert fb.shape == (N_MELS, 257) and (fb.sum(axis=1) > 0).all()
    peak_band = log_mel(tone(1.2, freq=1000.0)).mean(axis=1).argmax()
    assert 10 < peak_band < 25                     # 1 kHz sits in the lower-middle mel bands


def test_speech_span_finds_the_loud_part():
    s, e = speech_span(clip(("silence", 0.4), ("speech", 0.6), ("silence", 0.5)))
    assert abs(s / SR - 0.4) < 0.03 and abs(e / SR - 1.0) < 0.03
    assert speech_span(clip(("silence", 1.0))) is None


def test_word_span_stops_at_the_pause_before_a_command():
    audio = clip(("silence", 0.3), ("speech", 0.6), ("silence", 0.2), ("speech", 1.2), ("silence", 0.3))
    s, e = word_span(audio)
    assert abs(s / SR - 0.3) < 0.03 and abs(e / SR - 0.9) < 0.05


def test_positive_windows_hold_the_whole_word():
    audio = clip(("silence", 0.5), ("speech", 0.6), ("silence", 0.5))
    rng = np.random.default_rng(1)
    for w in positive_windows(audio, 20, rng):
        assert len(w) == WINDOW
        assert np.sum(np.abs(w) > 0.1) >= 0.55 * 0.6 * SR    # nearly all of the word's loud samples


def test_partial_windows_end_inside_the_word():
    audio = clip(("silence", 0.5), ("speech", 0.6), ("silence", 0.5))
    for w in partial_windows(audio, 20, np.random.default_rng(2)):
        loud = np.sum(np.abs(w) > 0.1)
        assert 0 < loud < 0.45 * 0.6 * SR


def test_augmentation_keeps_the_window_length():
    rng = np.random.default_rng(3)
    aug = Augmenter([np.random.default_rng(4).standard_normal(SR).astype(np.float32)], rng)
    for w in random_windows(clip(("speech", 2.0)), 10, rng):
        assert len(_fit(aug(w))) == WINDOW


def test_detector_streams_an_exported_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import json

    from aiforme.wakeword.detector import HOP, Detector
    from aiforme.wakeword.train import Normalized, make_model

    class Loud(torch.nn.Module):          # "wake word" = loud audio, to test the streaming logic only
        def forward(self, x):
            return x.amax(dim=(1, 2, 3))[:, None] * 4

    path = tmp_path / "ww.onnx"
    model = Normalized(Loud(), np.zeros(N_MELS, np.float32), np.ones(N_MELS, np.float32))
    torch.onnx.export(model, torch.zeros(1, 1, N_MELS, FRAMES), str(path), input_names=["features"],
                      output_names=["logit"], dynamo=False)
    (tmp_path / "ww.json").write_text(json.dumps({"threshold": 0.5, "consecutive": 2, "refractory_s": 3.0}))
    det = Detector(path, tmp_path / "ww.json")
    assert det.count(clip(("silence", 3.0))) == 0
    assert det.count(clip(("silence", 1.0), ("speech", 1.5), ("silence", 1.0))) == 1   # once, then refractory
    assert make_model()(torch.zeros(2, 1, N_MELS, FRAMES)).shape == (2, 1)
    assert HOP == 1280

    # the fast whole-recording scoring used for thresholds and mining matches the streaming detector
    from aiforme.wakeword.train import count_hits, stream_probs

    class Ramp(torch.nn.Module):          # smooth scores, so a mismatch would show in the values
        def forward(self, x):
            return x.mean(dim=(1, 2, 3))[:, None]

    path2 = tmp_path / "ramp.onnx"
    zeros, ones = np.zeros(N_MELS, np.float32), np.ones(N_MELS, np.float32)
    torch.onnx.export(Normalized(Ramp(), zeros, ones), torch.zeros(1, 1, N_MELS, FRAMES), str(path2),
                      input_names=["features"], output_names=["logit"], dynamo=False)
    audio = clip(("silence", 0.7), ("speech", 0.9), ("silence", 1.1), ("speech", 0.4))
    det2 = Detector(path2, tmp_path / "ww.json")
    expected = []
    for i in range(0, len(audio) - HOP + 1, HOP):
        det2._buffer = np.concatenate([det2._buffer[HOP:], audio[i:i + HOP]])
        expected.append(det2.probability(det2._buffer))
    fast = stream_probs(Ramp(), zeros, ones, audio)
    assert np.allclose(fast, expected, atol=1e-5)
    probs = stream_probs(Loud(), zeros, ones, clip(("silence", 1.0), ("speech", 1.5), ("silence", 1.0)))
    assert count_hits(probs, 0.5, refractory_s=3.0) == 1 and count_hits(probs, 0.5) == 2
