import threading

import pytest

from aiforme.core import ERROR_TOPIC, Event, EventBus, Module, run, run_once


class Echo(Module):
    name = "echo"
    subscribes = ("ping.*",)

    def handle(self, event):
        if event.payload.get("fail"):
            raise RuntimeError("asked to fail")
        if event.payload.get("silent"):
            return None
        yield Event("pong.sent", {"reply_to": event.id})


@pytest.fixture
def bus(tmp_path):
    with EventBus(tmp_path / "bus.sqlite3") as b:
        b.subscribe("echo", Echo.subscribes)
        b.subscribe("watcher", ["*"])
        yield b


def test_outputs_are_published_with_the_module_as_source(bus):
    ping = bus.publish(Event("ping.sent"))
    assert run_once(bus, Echo()) == 1
    pong = bus.recent(pattern="pong.*")[-1]
    assert pong.payload == {"reply_to": ping.id}
    assert pong.source == "echo"
    assert run_once(bus, Echo()) == 0


def test_a_module_may_return_nothing(bus):
    bus.publish(Event("ping.sent", {"silent": True}))
    assert run_once(bus, Echo()) == 1
    assert bus.recent(pattern="pong.*") == []


def test_a_failing_event_becomes_an_error_and_does_not_block_the_queue(bus):
    bad = bus.publish(Event("ping.sent", {"fail": True}))
    bus.publish(Event("ping.sent"))
    assert run_once(bus, Echo()) == 2
    [error] = bus.recent(pattern="system.*")
    assert error.topic == ERROR_TOPIC
    assert error.payload["event_id"] == bad.id
    assert "asked to fail" in error.payload["error"]
    assert len(bus.recent(pattern="pong.*")) == 1


def test_run_until_stopped(tmp_path):
    path = tmp_path / "bus.sqlite3"
    stop = threading.Event()
    started = threading.Event()

    def worker():                            # a sqlite connection belongs to the thread that opened it
        with EventBus(path) as bus:
            bus.subscribe(Echo.name, Echo.subscribes)
            started.set()
            run(bus, [Echo()], interval=0.01, stop=stop)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert started.wait(5)
        with EventBus(path) as client:
            client.subscribe("client", ["pong.*"])
            client.publish(Event("ping.sent"))
            for _ in range(500):
                if client.poll("client"):
                    break
                stop.wait(0.01)
            assert client.poll("client"), "no pong within 5 s"
    finally:
        stop.set()
        thread.join(timeout=5)


def test_module_names_must_be_unique(bus):
    with pytest.raises(ValueError):
        run(bus, [Echo(), Echo()])
