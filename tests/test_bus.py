import pytest

from aiforme.core import Event, EventBus


@pytest.fixture
def bus(tmp_path):
    with EventBus(tmp_path / "bus.sqlite3") as b:
        yield b


def test_event_rejects_bad_topics():
    for topic in ["", "Voice.x", "voice..x", "voice.", "voice x", "voice.*"]:
        with pytest.raises(ValueError):
            Event(topic)
    with pytest.raises(TypeError):
        Event("voice.transcribed", payload=["not", "a", "dict"])


def test_publish_assigns_increasing_ids(bus):
    a = bus.publish(Event("voice.transcribed", {"text": "привет"}))
    b = bus.publish(Event("voice.transcribed"))
    assert a.id is not None and b.id > a.id
    assert bus.recent()[0].payload == {"text": "привет"}


def test_new_consumer_starts_at_the_end(bus):
    bus.publish(Event("voice.transcribed", {"n": 1}))
    bus.subscribe("late", ["voice.*"])
    assert bus.poll("late") == []
    bus.publish(Event("voice.transcribed", {"n": 2}))
    assert [e.payload["n"] for e in bus.poll("late")] == [2]


def test_from_start_replays_history(bus):
    bus.publish(Event("voice.transcribed", {"n": 1}))
    bus.subscribe("replay", ["voice.transcribed"], from_start=True)
    assert [e.payload["n"] for e in bus.poll("replay")] == [1]


@pytest.mark.parametrize(("pattern", "expected"), [
    ("voice.transcribed", ["voice.transcribed"]),
    ("voice.*", ["voice.transcribed", "voice.wake.detected"]),
    ("*", ["voice", "voice.transcribed", "voice.wake.detected", "voicemail.received"]),
])
def test_patterns(bus, pattern, expected):
    bus.subscribe("c", [pattern])
    for topic in ["voice", "voice.transcribed", "voice.wake.detected", "voicemail.received"]:
        bus.publish(Event(topic))
    assert [e.topic for e in bus.poll("c")] == expected


def test_bad_pattern_is_rejected(bus):
    for pattern in ["voice*", "voice.[a]", "*.x", ""]:
        with pytest.raises(ValueError):
            bus.subscribe("c", [pattern])


def test_event_stays_pending_until_completed(bus):
    bus.subscribe("c", ["a.*"])
    event = bus.publish(Event("a.x"))
    assert bus.poll("c") == [event]
    assert bus.poll("c") == [event]          # not completed: delivered again
    bus.complete("c", event)
    assert bus.poll("c") == []


def test_complete_publishes_outputs_and_skips_unrelated_events(bus):
    bus.subscribe("c", ["a.*"])
    bus.subscribe("watcher", ["b.*"])
    event = bus.publish(Event("a.x"))
    bus.publish(Event("z.unrelated"))
    [out] = bus.complete("c", event, [Event("b.y", {"from": event.id})])
    assert bus.poll("watcher") == [out]
    assert bus.poll("c") == []


def test_complete_is_atomic(bus):
    bus.subscribe("c", ["a.*"])
    event = bus.publish(Event("a.x"))
    with pytest.raises(TypeError):           # the payload is not JSON-serialisable
        bus.complete("c", event, [Event("b.ok"), Event("b.bad", {"x": object()})])
    assert bus.recent(pattern="b.*") == []
    assert bus.poll("c") == [event]          # the cursor did not move either


def test_cursors_are_per_consumer(bus):
    bus.subscribe("one", ["a.*"])
    bus.subscribe("two", ["a.*"])
    event = bus.publish(Event("a.x"))
    bus.complete("one", event)
    assert bus.poll("one") == []
    assert bus.poll("two") == [event]


def test_resubscribe_keeps_the_cursor(bus):
    bus.subscribe("c", ["a.*"])
    event = bus.publish(Event("a.x"))
    bus.subscribe("c", ["a.*", "b.*"], from_start=True)
    assert bus.poll("c") == [event]


def test_unknown_consumer(bus):
    with pytest.raises(KeyError):
        bus.poll("nobody")
    event = bus.publish(Event("a.x"))
    with pytest.raises(KeyError):
        bus.complete("nobody", event)


def test_two_processes_share_one_file(tmp_path):
    path = tmp_path / "bus.sqlite3"
    with EventBus(path) as producer, EventBus(path) as consumer:
        consumer.subscribe("c", ["a.*"])
        event = producer.publish(Event("a.x", {"n": 1}))
        assert consumer.poll("c") == [event]
        consumer.complete("c", event)
    with EventBus(path) as reopened:         # state survives a restart
        assert reopened.poll("c") == []
        assert [e.payload for e in reopened.recent()] == [{"n": 1}]
