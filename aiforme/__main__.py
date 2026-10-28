"""Command line.

    python -m aiforme run                       # the voice assistant: wake word, speech, actions
    python -m aiforme run --push-to-talk        # Enter instead of «Амадеус»
    python -m aiforme run --no-voice            # no microphone; send commands with `say`
    python -m aiforme say "открой телеграм"     # a typed command, as if it was spoken
    python -m aiforme dashboard                 # only the web UI, http://127.0.0.1:8765
    python -m aiforme retrain                   # the intent classifier on the commands labelled in the UI
    python -m aiforme publish voice.transcribed '{"text": "open the terminal"}'
    python -m aiforme tail -n 50 'voice.*'
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from datetime import datetime, timedelta

from aiforme.core import Event, EventBus, run
from aiforme.core.config import activity_path, bus_path, calendar_path, feedback_path, modes_path

EMBEDDERS = ("tfidf", "charcnn", "minilm", "e5")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiforme")
    commands = parser.add_subparsers(dest="command", required=True)
    pub = commands.add_parser("publish", help="publish one event")
    pub.add_argument("topic")
    pub.add_argument("payload", nargs="?", default="{}", help="JSON object")
    tail = commands.add_parser("tail", help="show the last events")
    tail.add_argument("pattern", nargs="?", default="*")
    tail.add_argument("-n", type=int, default=20)
    say = commands.add_parser("say", help="send a typed command to a running assistant")
    say.add_argument("text")
    start = commands.add_parser("run", help="run the voice assistant")
    start.add_argument("--embedder", default="e5", choices=EMBEDDERS)
    start.add_argument("--no-voice", action="store_true", help="no microphone, typed commands only")
    start.add_argument("--push-to-talk", action="store_true", help="Enter instead of the wake word")
    start.add_argument("--no-speak", action="store_true", help="print replies without speaking them")
    start.add_argument("--no-duck", action="store_true", help="do not turn the volume down while listening")
    start.add_argument("--stt-device", default="auto", choices=("auto", "cuda", "cpu"))
    start.add_argument("--input-device", default=None, help="microphone index or name (sounddevice)")
    dash = commands.add_parser("dashboard", help="run only the web UI")
    dash.add_argument("--embedder", default="e5", choices=EMBEDDERS, help="what the Retrain button trains")
    learn = commands.add_parser("retrain", help="retrain the intent classifier on the labelled commands")
    learn.add_argument("--embedder", default="e5", choices=EMBEDDERS)
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run(args)
    if args.command == "dashboard":
        return _dashboard(args.embedder)
    if args.command == "retrain":
        return _retrain(args.embedder)
    with EventBus(bus_path()) as bus:
        if args.command == "publish":
            event = bus.publish(Event(args.topic, json.loads(args.payload), source="cli"))
            print(f"published #{event.id} {event.topic}")
        elif args.command == "say":
            bus.publish(Event("text.command", {"text": args.text}, source="cli"))
        else:
            for event in bus.recent(args.n, args.pattern):
                when = datetime.fromtimestamp(event.created_at).strftime("%H:%M:%S")
                payload = json.dumps(event.payload, ensure_ascii=False)
                print(f"#{event.id} {when} {event.topic} [{event.source}] {payload}")
    return 0


def _run(args) -> int:
    from aiforme.active_learning.retrain import current_model
    from aiforme.active_learning.store import Feedback
    from aiforme.voice_orchestrator.pipeline import ActionModule, IntentModule, ReplyModule, windows_speak
    from aiforme.voice_orchestrator.windows import WindowsDesktop

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"Training the intent classifier ({args.embedder})...", flush=True)
    from aiforme.calendar import Calendar
    calendar = Calendar(calendar_path())

    def reload():
        with Feedback(feedback_path()) as feedback:
            return current_model(feedback, args.embedder)

    classifier, version = reload()
    print(f"Intent model version {version}.", flush=True)
    modules = [IntentModule(classifier, version, reload), ActionModule(WindowsDesktop(), agenda=_today(calendar)),
               ReplyModule(None if args.no_speak else windows_speak)]
    stop = threading.Event()

    def serve():                # a sqlite connection belongs to the thread that opened it
        with EventBus(bus_path()) as bus:
            run(bus, modules, interval=0.05, stop=stop)

    worker = threading.Thread(target=serve, name="modules")
    worker.start()
    background = _background(stop, calendar, args.embedder)
    try:
        if args.no_voice:
            print('Ready. Send commands with: python -m aiforme say "open notepad"', flush=True)
            stop.wait()
        else:
            _listen(args, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join()
        for thread in background:
            thread.join(timeout=5)
    return 0


def _today(calendar):
    def agenda(now: datetime):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return calendar.between(start, start + timedelta(days=1))
    return agenda


def _background(stop: threading.Event, calendar, embedder: str) -> list[threading.Thread]:
    """The web UI, the calendar watcher and the mode autopilot, in threads of their own, until `stop`."""
    from aiforme.calendar.watcher import CalendarWatcher
    from aiforme.dashboard.server import HOST, PORT, serve
    from aiforme.mode_autopilot.actions import ModeActions, WindowsSystem, load_settings
    from aiforme.mode_autopilot.activity import WindowsActivity
    from aiforme.mode_autopilot.autopilot import Autopilot
    from aiforme.mode_autopilot.log import ActivityLog

    server = serve(calendar, bus_path(), feedback_path(), embedder)

    def ui():
        threading.Thread(target=lambda: (stop.wait(), setattr(server, "should_exit", True)), daemon=True).start()
        server.run()

    def watch():
        with EventBus(bus_path()) as bus:
            CalendarWatcher(calendar).run(bus, stop)

    def autopilot():
        actions = ModeActions(WindowsSystem(), load_settings(modes_path()))
        pilot = Autopilot(WindowsActivity().sample, actions, activity_log=ActivityLog(activity_path()))
        with EventBus(bus_path()) as bus:
            pilot.run(bus, stop)

    threads = [threading.Thread(target=ui, name="dashboard"), threading.Thread(target=watch, name="calendar"),
               threading.Thread(target=autopilot, name="autopilot")]
    for thread in threads:
        thread.start()
    print(f"Dashboard: http://{HOST}:{PORT}", flush=True)
    return threads


def _dashboard(embedder: str) -> int:
    from aiforme.calendar import Calendar
    from aiforme.dashboard.server import HOST, PORT, WEB_DIST, serve

    if not WEB_DIST.exists():
        print("The UI is not built yet: cd aiforme/dashboard/web && npm install && npm run build", file=sys.stderr)
    print(f"Dashboard: http://{HOST}:{PORT}  (Ctrl+C to stop)", flush=True)
    serve(Calendar(calendar_path()), bus_path(), feedback_path(), embedder).run()
    return 0


def _retrain(embedder: str) -> int:
    from aiforme.active_learning.retrain import NothingToLearn, retrain_and_announce
    from aiforme.active_learning.store import Feedback

    with Feedback(feedback_path()) as feedback, EventBus(bus_path()) as bus:
        try:
            v = retrain_and_announce(feedback, bus, embedder)
        except NothingToLearn:
            print("Nothing to learn: label some commands in the dashboard first.", file=sys.stderr)
            return 1
    m = v.metrics
    print(f"version {v.id}: {'accepted' if v.accepted else 'rejected'}, {m['real_labels']} new labels")
    print(f"  test phrases         {m['test_before']['accuracy']:.1%} -> {m['test']['accuracy']:.1%}")
    print(f"  the labelled commands {m['real_before']['accuracy']:.1%} -> {m['real']['accuracy']:.1%} (out of fold)")
    return 0


def _listen(args, stop: threading.Event) -> None:
    from aiforme.voice_orchestrator.listener import EnterKey, Listener, SileroVAD, WhisperSTT
    from aiforme.wakeword.detector import Detector

    print("Loading Whisper...", flush=True)
    stt = WhisperSTT(device=args.stt_device)
    print(f"Whisper {stt.model_name} on {stt.device}.", flush=True)
    wake = EnterKey() if args.push_to_talk else Detector()
    device = int(args.input_device) if (args.input_device or "").isdigit() else args.input_device
    vad = SileroVAD()
    ducker = None
    if not args.no_duck:
        try:
            from aiforme.voice_orchestrator.media import VolumeDucker
            ducker = VolumeDucker()
        except Exception as exc:                    # no audio device or no pycaw: listen without ducking
            print(f"Volume ducking off: {exc}", file=sys.stderr)
    with EventBus(bus_path()) as bus:
        Listener(bus, wake, stt, input_device=device, vad=vad, ducker=ducker).run(stop)


if __name__ == "__main__":
    sys.exit(main())
