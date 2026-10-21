"""Command line for inspecting the bus by hand.

    python -m aiforme publish voice.transcribed '{"text": "open the terminal"}'
    python -m aiforme tail -n 50 'voice.*'
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from aiforme.core import Event, EventBus
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
    args = parser.parse_args(argv)

    with EventBus(bus_path()) as bus:
        if args.command == "publish":
            event = bus.publish(Event(args.topic, json.loads(args.payload), source="cli"))
            print(f"published #{event.id} {event.topic}")
        else:
            for event in bus.recent(args.n, args.pattern):
                when = datetime.fromtimestamp(event.created_at).strftime("%H:%M:%S")
                payload = json.dumps(event.payload, ensure_ascii=False)
                print(f"#{event.id} {when} {event.topic} [{event.source}] {payload}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
