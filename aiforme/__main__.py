"""Command line.

    python -m aiforme run                       # the voice assistant: wake word, speech, actions
    python -m aiforme run --push-to-talk        # Enter instead of «Амадеус»
    python -m aiforme run --no-voice            # no microphone; send commands with `say`
    python -m aiforme say "открой телеграм"     # a typed command, as if it was spoken
    python -m aiforme publish voice.transcribed '{"text": "open the terminal"}'
    python -m aiforme tail -n 50 'voice.*'
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from datetime import datetime

from aiforme.core import Event, EventBus, run
from aiforme.core.config import bus_path


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
    start.add_argument("--embedder", default="e5", choices=("tfidf", "charcnn", "minilm", "e5"))
    start.add_argument("--no-voice", action="store_true", help="no microphone, typed commands only")
    start.add_argument("--push-to-talk", action="store_true", help="Enter instead of the wake word")
    start.add_argument("--no-speak", action="store_true", help="print replies without speaking them")
    start.add_argument("--stt-device", default="auto", choices=("auto", "cuda", "cpu"))
    start.add_argument("--input-device", default=None, help="microphone index or name (sounddevice)")
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run(args)
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
    from aiforme.voice_orchestrator.nlu import train
    from aiforme.voice_orchestrator.pipeline import ActionModule, IntentModule, ReplyModule, windows_speak
    from aiforme.voice_orchestrator.windows import WindowsDesktop

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"Training the intent classifier ({args.embedder})...", flush=True)
    modules = [IntentModule(train(args.embedder)), ActionModule(WindowsDesktop()),
               ReplyModule(None if args.no_speak else windows_speak)]
    stop = threading.Event()

    def serve():                # a sqlite connection belongs to the thread that opened it
        with EventBus(bus_path()) as bus:
            run(bus, modules, interval=0.05, stop=stop)

    worker = threading.Thread(target=serve, name="modules")
    worker.start()
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
    return 0


def _listen(args, stop: threading.Event) -> None:
    from aiforme.voice_orchestrator.listener import EnterKey, Listener, WhisperSTT
    from aiforme.wakeword.detector import Detector

    print("Loading Whisper...", flush=True)
    stt = WhisperSTT(device=args.stt_device)
    print(f"Whisper {stt.model_name} on {stt.device}.", flush=True)
    wake = EnterKey() if args.push_to_talk else Detector()
    device = int(args.input_device) if (args.input_device or "").isdigit() else args.input_device
    with EventBus(bus_path()) as bus:
        Listener(bus, wake, stt, input_device=device).run(stop)


if __name__ == "__main__":
    sys.exit(main())
