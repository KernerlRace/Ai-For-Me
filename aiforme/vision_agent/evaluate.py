"""Scores a grounder on the collected snapshots: does its click land inside the element the instruction names?

    python -m aiforme vision-eval                  # the UI Automation baseline
    python -m aiforme vision-eval --grounder uia

Prints accuracy overall, by app and by control type, and writes every task's outcome to
data/vision/reports/<grounder>-<time>.json. Reports stay in the data directory, never in the repository: the
instructions are names from my own windows — chats, files, tabs.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from aiforme.vision_agent.dataset import Store, tasks
from aiforme.vision_agent.grounding import Grounder


def evaluate(grounder: Grounder, store: Store) -> dict:
    snapshots = {s.id: s for s in store.snapshots()}
    outcomes = []
    for task in tasks(snapshots.values()):
        image = store.image(task.snapshot)
        png = image.read_bytes() if image.exists() else None
        t0 = time.perf_counter()
        point = grounder.locate(snapshots[task.snapshot], png, task.instruction)
        seconds = time.perf_counter() - t0
        outcomes.append({**asdict(task), "point": point, "hit": task.hit(point), "seconds": seconds})
    return {"grounder": grounder.name, "snapshots": len(snapshots), **summarize(outcomes), "tasks": outcomes}


def summarize(outcomes: list[dict]) -> dict:
    """Accuracy (a hit counts), how often it gave no answer at all, latency, and accuracy per app and per type."""
    def score(rows):
        n = len(rows)
        return {"n": n, "accuracy": sum(r["hit"] for r in rows) / n if n else 0.0,
                "no_answer": sum(r["point"] is None for r in rows) / n if n else 0.0}

    by = {"app": defaultdict(list), "type": defaultdict(list)}
    for r in outcomes:
        by["app"][r["app"]].append(r)
        by["type"][r["type"]].append(r)
    seconds = sorted(r["seconds"] for r in outcomes)
    return {
        **score(outcomes),
        "median_seconds": seconds[len(seconds) // 2] if seconds else None,
        "by_app": {k: score(v) for k, v in sorted(by["app"].items())},
        "by_type": {k: score(v) for k, v in sorted(by["type"].items())},
    }


def report(result: dict) -> str:
    lines = [f"{result['grounder']}: {result['accuracy']:.1%} of {result['n']} tasks from {result['snapshots']} "
             f"snapshots; no answer {result['no_answer']:.1%}; median {result['median_seconds'] or 0:.3f}s"]
    for group in ("by_app", "by_type"):
        lines.append(f"  {group[3:]}:")
        lines += [f"    {k:28} {v['accuracy']:6.1%} of {v['n']}" for k, v in result[group].items()]
    return "\n".join(lines)


def save(result: dict, root: Path) -> Path:
    out = root / "reports" / f"{result['grounder']}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
