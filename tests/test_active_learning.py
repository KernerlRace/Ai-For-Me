import time

import pytest
from fastapi.testclient import TestClient

from aiforme.active_learning.retrain import MODEL_UPDATED, NothingToLearn, current_model, retrain, training_set
from aiforme.active_learning.store import Feedback, LabelLocked
from aiforme.calendar import Calendar
from aiforme.core import Event, EventBus
from aiforme.dashboard.server import create_app
from aiforme.voice_orchestrator.nlu import TEST, TRAIN, IntentClassifier, TfidfEmbedder, load_examples
from aiforme.voice_orchestrator.pipeline import IntentModule

# slang the hand-written commands never use: the base classifier gets all of them wrong
SLANG = [(f"раскочегарь {app}", "open_app", {"app": app})
         for app in ("телегу", "дискорд", "стим", "хром", "спотифай", "телеграм")]


def heard(fb, event_id, text, intent, slots, predicted="out_of_scope", served_by=0):
    return fb.label(event_id, text, predicted, {}, 0.2, served_by, time.time(), intent, slots)


def test_labels_are_pending_until_a_version_takes_them():
    with Feedback() as fb:
        right = fb.label(1, "открой хром", "open_app", {"app": "хром"}, 0.9, 0, 1.0, "open_app", {"app": "хром"})
        wrong = heard(fb, 2, "врубай телегу", "open_app", {"app": "телегу"})
        assert (right.verdict, wrong.verdict) == ("confirmed", "corrected")
        heard(fb, 2, "врубай телегу", "open_app", {"app": "телегу"}, served_by=0)     # a pending label can change
        assert [lab.event_id for lab in fb.labels(pending=True)] == [1, 2]

        v = fb.add_version("tfidf", True, fb.labels(pending=True), 300, {})
        assert fb.current_version() == v.id and fb.labels(pending=True) == []
        with pytest.raises(LabelLocked):
            heard(fb, 2, "врубай телегу", "mute", {})
        with pytest.raises(LabelLocked):
            fb.remove(1)
        fb.add_version("tfidf", False, [heard(fb, 3, "тише", "volume_down", {})], 301, {})
        assert fb.current_version() == v.id                # a rejected version is never current
        assert fb.labels(pending=True)[0].event_id == 3
        assert fb.live_accuracy() == [{"model": 0, "labelled": 3, "correct": 1}]


def test_training_set_keeps_the_test_phrases_out():
    base, test = load_examples(TRAIN), load_examples(TEST)
    with Feedback() as fb:
        heard(fb, 1, base[0].text.upper(), "mute", {})                 # relabels a hand-written command
        heard(fb, 2, test[0].text, test[0].intent, test[0].slots)      # a test phrase said for real
        heard(fb, 3, "врубай телегу", "open_app", {"app": "телегу"})
        data = training_set(fb.labels(), base, test)
    assert len(data) == len(base) + 1
    assert [e.intent for e in data if e.text == base[0].text.upper()] == ["mute"]
    assert not any(e.text == test[0].text for e in data)


def test_retraining_learns_the_slang_and_the_assistant_reloads():
    with Feedback() as fb:
        with pytest.raises(NothingToLearn):
            retrain(fb, "tfidf")
        for i, (text, intent, slots) in enumerate(SLANG, 1):
            heard(fb, i, text, intent, slots)
        v = retrain(fb, "tfidf")
        assert v.accepted and v.id == 1 and v.new_labels == len(SLANG)
        assert v.metrics["real_before"]["accuracy"] == 0.0 < v.metrics["real"]["accuracy"]
        assert v.metrics["test_phrases_lost"] <= 1
        assert [x.id for x in fb.versions()] == [0, 1]     # version 0 recorded with the first retraining

        old = IntentClassifier(TfidfEmbedder()).fit(load_examples(TRAIN))
        module = IntentModule(old, 0, reload=lambda: current_model(fb, "tfidf"))
        assert next(iter(module.handle(Event("text.command", {"text": "раскочегарь вотсап"})))).payload["model"] == 0
        assert list(module.handle(Event(MODEL_UPDATED, {"version": 1}))) == []
        out = next(iter(module.handle(Event("text.command", {"text": "раскочегарь вотсап"})))).payload
        assert (out["intent"], out["slots"], out["model"]) == ("open_app", {"app": "вотсап"}, 1)


def test_labels_that_break_the_test_set_are_rejected():
    with Feedback() as fb:
        # "open X" relabelled as mute, over and over: consistent with itself, wrong on the test phrases like it
        apps = ["хром", "стим", "дискорд", "телеграм", "спотифай", "блокнот", "терминал", "зум"]
        for i, text in enumerate(f"{verb} {app}" for verb in ("открой", "запусти") for app in apps):
            heard(fb, i + 1, text, "mute", {}, predicted="open_app")
        v = retrain(fb, "tfidf")
        assert not v.accepted and not v.metrics["test_ok"] and v.metrics["test_phrases_lost"] > 1
        assert fb.current_version() == 0 and len(fb.labels(pending=True)) == 16


# --- the dashboard API --------------------------------------------------------------------------------------

def make(tmp_path):
    bus_path, fb_path = tmp_path / "bus.sqlite3", tmp_path / "feedback.sqlite3"
    with EventBus(bus_path) as bus:
        bus.subscribe("watch", ["intent.model_updated"])
        ids = [bus.publish(Event(t, p)).id for t, p in [
            ("intent.recognized", {"text": "открой хром", "intent": "open_app", "confidence": 0.93,
                                   "slots": {"app": "хром"}, "model": 0}),
            ("intent.unknown", {"text": "врубай телегу", "intent": "out_of_scope", "confidence": 0.11,
                                "slots": {}, "best_guess": "open_app", "model": 0}),
            ("action.done", {"intent": "open_app"}),
        ]]
    client = TestClient(create_app(Calendar(tmp_path / "calendar.sqlite3"), bus_path, fb_path, "tfidf"))
    return client, ids, bus_path


def test_commands_are_listed_newest_first_with_their_labels(tmp_path):
    client, (chrome, slang, action), _ = make(tmp_path)
    r = client.put(f"/api/commands/{chrome}/label", json={"intent": "open_app", "slots": {"app": "хром"}})
    assert r.status_code == 200 and r.json()["verdict"] == "confirmed"
    cmds = client.get("/api/commands").json()
    assert [c["id"] for c in cmds] == [slang, chrome]
    assert cmds[0]["intent"] == "out_of_scope" and cmds[0]["best_guess"] == "open_app" and cmds[0]["label"] is None
    assert cmds[1]["label"]["verdict"] == "confirmed"
    assert client.delete(f"/api/commands/{chrome}/label").status_code == 204
    assert client.delete(f"/api/commands/{chrome}/label").status_code == 404


def test_bad_labels_are_rejected(tmp_path):
    client, (chrome, slang, action), _ = make(tmp_path)
    put = lambda eid, body: client.put(f"/api/commands/{eid}/label", json=body).status_code  # noqa: E731
    assert put(action, {"intent": "open_app", "slots": {"app": "хром"}}) == 404       # not a command
    assert put(slang, {"intent": "order_pizza"}) == 422
    assert put(slang, {"intent": "open_app"}) == 422                                  # the app is missing
    assert put(slang, {"intent": "open_app", "slots": {"app": "телеграм"}}) == 422    # not words of the command
    r = client.put(f"/api/commands/{slang}/label", json={"intent": "mute", "slots": {"app": "телегу"}})
    assert r.status_code == 200 and r.json()["slots"] == {}                           # mute takes no slot


def test_retrain_from_the_page(tmp_path):
    client, (chrome, slang, _), bus_path = make(tmp_path)
    assert client.post("/api/learning/retrain").status_code == 409                    # nothing labelled yet
    with Feedback(tmp_path / "feedback.sqlite3") as fb:
        for i, (text, intent, slots) in enumerate(SLANG, 100):
            heard(fb, i, text, intent, slots)
    assert client.put(f"/api/commands/{slang}/label",
                      json={"intent": "open_app", "slots": {"app": "телегу"}}).status_code == 200
    assert client.get("/api/learning").json()["pending"] == {"confirmed": 0, "corrected": len(SLANG) + 1}

    assert client.post("/api/learning/retrain").status_code == 202
    client.app.state.retraining["thread"].join(timeout=120)
    state = client.get("/api/learning").json()
    assert state["error"] is None and not state["retraining"]
    assert state["version"] == 1 and [v["accepted"] for v in state["versions"]] == [True, True]
    assert state["pending"] == {"confirmed": 0, "corrected": 0}
    with EventBus(bus_path) as bus:
        assert [e.payload for e in bus.poll("watch")] == [{"version": 1}]
    assert client.put(f"/api/commands/{slang}/label", json={"intent": "mute"}).status_code == 409


def test_bus_get(tmp_path):
    with EventBus(tmp_path / "bus.sqlite3") as bus:
        e = bus.publish(Event("ping.sent", {"n": 1}))
        assert bus.get(e.id).payload == {"n": 1} and bus.get(e.id + 1) is None
