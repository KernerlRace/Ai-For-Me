from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from aiforme.active_learning.store import Feedback
from aiforme.calendar import Calendar
from aiforme.dashboard.server import create_app
from aiforme.demo import DEMOS, DemoExists, build
from aiforme.voice_orchestrator.nlu import normalize


@pytest.mark.parametrize("lang", sorted(DEMOS))
def test_demo_fills_every_page_through_the_real_pipeline(tmp_path, lang):
    FIRST, SECOND, WEEK = DEMOS[lang]
    summary = build(tmp_path, "tfidf", today=datetime(2026, 11, 20, 14), log=lambda *_: None, lang=lang)

    with Calendar(tmp_path / "calendar.sqlite3") as calendar:
        week = calendar.between(datetime(2026, 11, 16), datetime(2026, 11, 23))
        assert len(week) == len(WEEK)
        assert {e.kind for e in week} == {"meeting", "shift", "other"}
        client = TestClient(create_app(calendar, tmp_path / "bus.sqlite3", tmp_path / "feedback.sqlite3", "tfidf",
                                       auto_retrain_every=0))
        assert client.get("/api/mode").json()["mode"] == "coding"
        commands = client.get("/api/commands", params={"limit": 500}).json()
        assert len(commands) == len(FIRST) + len(SECOND)
        assert any(c["label"] is None for c in commands)            # left for "Uncertain first"
        learning = client.get("/api/learning").json()

    # version 0 and the first retraining, whatever the gate decided
    assert [v["id"] for v in learning["versions"]] == [0, 1] == [0] + [v["id"] for v in summary["versions"]]
    with Feedback(tmp_path / "feedback.sqlite3") as fb:
        served = {row["model"] for row in fb.live_accuracy()}
    assert 0 in served and summary["pending"] > 0


def test_demo_never_writes_over_a_bus(tmp_path):
    (tmp_path / "bus.sqlite3").write_bytes(b"")
    with pytest.raises(DemoExists):
        build(tmp_path, "tfidf", log=lambda *_: None)


def test_every_demo_label_is_one_the_retraining_accepts():
    # a labelled slot must be words of its command, or retraining on that label would refuse it
    for lang, (first, second, _) in DEMOS.items():
        for text, _intent, slot in first + second:
            if slot is not None:
                assert f" {normalize(slot)} " in f" {normalize(text)} ", (lang, text, slot)
    with pytest.raises(ValueError):
        build("unused", "tfidf", lang="de")
