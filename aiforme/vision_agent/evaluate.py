"""Scores a grounder on the collected snapshots: does its click land inside the element the instruction names?

    python -m aiforme vision-eval                  # the UI Automation baseline
    python -m aiforme vision-eval --grounder vlm  # Qwen2.5-VL through Ollama
    python -m aiforme vision-eval --grounder vlm --model qwen2.5vl:3b --limit 20
    python -m aiforme vision-eval --tasks described   # commands in my own words, not the element's name

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

from aiforme.vision_agent.dataset import Store, described_tasks, tasks
from aiforme.vision_agent.grounding import Grounder


def evaluate(grounder: Grounder, store: Store, limit: int | None = None, task_set: str = "literal") -> dict:
    """`task_set` is "literal" (the instruction is the element's name) or "described" (my own words, from
    described.tsv). `limit` takes every k-th task so that a quick run still covers every snapshot."""
    snapshots = {s.id: s for s in store.snapshots()}
    if task_set == "literal":
        chosen = tasks(snapshots.values())
    elif task_set == "described":
        chosen = described_tasks(snapshots.values(), store.described)
    else:
        raise ValueError(f"unknown task set {task_set!r}: expected literal or described")
    if limit and len(chosen) > limit:
        step = len(chosen) / limit
        chosen = [chosen[int(i * step)] for i in range(limit)]
    outcomes = []
    for task in chosen:
        image = store.image(task.snapshot)
        png = image.read_bytes() if image.exists() else None
        t0 = time.perf_counter()
        point = grounder.locate(snapshots[task.snapshot], png, task.instruction)
        seconds = time.perf_counter() - t0
        row = {**asdict(task), "point": point, "hit": task.hit(point), "seconds": seconds}
        if hasattr(grounder, "last_reply"):             # a model's raw answer: what to read when it misses
            row["reply"] = grounder.last_reply
        outcomes.append(row)
    result = {"grounder": grounder.name, "task_set": task_set, "snapshots": len(snapshots), **summarize(outcomes)}
    if getattr(grounder, "frames", None):              # the server's coordinate frame per image size, as measured
        result["frames"] = {f"{w}x{h}": list(scale) for (w, h), scale in grounder.frames.items()}
    return {**result, "tasks": outcomes}


def summarize(outcomes: list[dict]) -> dict:
    """Accuracy (a hit counts), how often it gave no answer at all, how often it clicked somewhere wrong, latency,
    and the same per app and per type."""
    def score(rows):
        n = len(rows)
        return {"n": n, "accuracy": sum(r["hit"] for r in rows) / n if n else 0.0,
                "no_answer": sum(r["point"] is None for r in rows) / n if n else 0.0,
                "wrong_click": sum(r["point"] is not None and not r["hit"] for r in rows) / n if n else 0.0}

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
    lines = [f"{result['grounder']} on {result.get('task_set', 'literal')} tasks: {result['accuracy']:.1%} "
             f"of {result['n']} tasks from {result['snapshots']} "
             f"snapshots; no answer {result['no_answer']:.1%}; wrong click {result.get('wrong_click', 0):.1%}; "
             f"median {result['median_seconds'] or 0:.3f}s"]
    for group in ("by_app", "by_type"):
        lines.append(f"  {group[3:]}:")
        lines += [f"    {k:28} {v['accuracy']:6.1%} of {v['n']}" for k, v in result[group].items()]
    return "\n".join(lines)


def save(result: dict, root: Path) -> Path:
    name = f"{result['grounder']}-{result.get('task_set', 'literal')}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out = root / "reports" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
