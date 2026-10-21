import queue
import threading

import numpy as np

from aiforme.core import EventBus
from aiforme.voice_orchestrator.listener import FRAME, FRAME_S, Endpointer, Listener


def frame(level):
    return np.full(FRAME, level, dtype=np.int16)


def test_endpointer_ends_after_silence():
    ep = Endpointer(silence_s=0.8, noise=100)
    ep.start()
    assert all(ep.push(2000) is None for _ in range(10))     # speaking
    results = [ep.push(100) for _ in range(round(0.8 / FRAME_S))]
    assert results[-1] == "end" and set(results[:-1]) == {None}


def test_endpointer_gives_up_without_speech():
    ep = Endpointer(no_speech_s=1.0, noise=100)
    ep.start()
    results = [ep.push(100) for _ in range(round(1.0 / FRAME_S))]
    assert results[-1] == "no_speech"


def test_endpointer_stops_long_dictation():
    ep = Endpointer(max_s=2.0, noise=100)
    ep.start()
    results = [ep.push(2000) for _ in range(round(2.0 / FRAME_S))]
    assert results[-1] == "max"


def test_noise_floor_raises_the_speech_level():
    ep = Endpointer(noise=100, ratio=3.0, min_rms=300)
    for _ in range(200):
        ep.observe_noise(1000)                                # a loud fan
    ep.start()
    assert ep.push(2000) is None and not ep._speech           # fan + a little is not speech


class WakeAfter:
    def __init__(self, n):
        self.n, self.resets = n, 0

    def detected(self, f):
        self.n -= 1
        return self.n == 0

    def reset(self):
        self.resets += 1


class FakeSTT:
    def __init__(self, text):
        self.text, self.audio = text, None

    def transcribe(self, audio):
        self.audio = audio
        return self.text, "ru"


def run_listener(tmp_path, stt, frames):
    with EventBus(tmp_path / "bus.sqlite3") as bus:
        q = queue.Queue()
        for f in frames:
            q.put(f)
        stop = threading.Event()
        wake = WakeAfter(3)
        listener = Listener(bus, wake, stt, Endpointer(silence_s=0.4, noise=100), beep=lambda f: None)
        threading.Timer(0.5, stop.set).start()
        listener.loop(q, stop)
        return [(e.topic, e.payload) for e in bus.recent()], wake


def test_wake_word_then_command_is_transcribed(tmp_path):
    stt = FakeSTT("Открой Telegram.")
    speech = [frame(3000)] * 10
    events, wake = run_listener(tmp_path, stt, [frame(100)] * 3 + speech + [frame(100)] * 10)
    assert [t for t, _ in events] == ["voice.wake.detected", "voice.transcribed"]
    assert events[1][1]["text"] == "Открой Telegram."
    assert len(stt.audio) >= len(speech) * FRAME
    assert wake.resets == 1


def test_silence_after_wake_word_times_out(tmp_path):
    stt = FakeSTT("unused")
    events, _ = run_listener(tmp_path, stt, [frame(100)] * 3 + [frame(100)] * 50)
    assert [t for t, _ in events] == ["voice.wake.detected", "voice.timeout"]
    assert stt.audio is None


def test_whisper_hallucinations_are_dropped(tmp_path):
    events, _ = run_listener(tmp_path, FakeSTT("Продолжение следует..."),
                             [frame(100)] * 3 + [frame(3000)] * 5 + [frame(100)] * 10)
    assert [t for t, _ in events] == ["voice.wake.detected", "voice.timeout"]
