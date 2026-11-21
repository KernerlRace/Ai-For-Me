"""Is one grounder really better than another on the same tasks, or is it the luck of a few dozen commands?

    python -m aiforme.vision_agent.compare data/vision/reports/A.json data/vision/reports/B.json

Each accuracy comes with a 95% Wilson interval. The two runs are paired task by task, and McNemar's exact test asks
whether the tasks only one of them got right lean to one side more than chance would put them: on 59 tasks, a
difference of 10 points can be noise, and the test says when it is not.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% interval for a proportion; unlike the normal approximation it stays inside [0, 1] at 0% or 100%."""
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Tasks only `a` got right, tasks only `b` got right, and the exact two-sided p-value of that split."""
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return only_a, only_b, min(1.0, 2 * tail)


def paired(a: dict, b: dict) -> tuple[list[bool], list[bool]]:
    """The hits of two reports on the tasks both ran, matched by snapshot and instruction."""
    hits_b = {(t["snapshot"], t["instruction"]): t["hit"] for t in b["tasks"]}
    keys = [(t["snapshot"], t["instruction"]) for t in a["tasks"] if (t["snapshot"], t["instruction"]) in hits_b]
    hits_a = {(t["snapshot"], t["instruction"]): t["hit"] for t in a["tasks"]}
    return [hits_a[k] for k in keys], [hits_b[k] for k in keys]


def compare(a: dict, b: dict) -> str:
    x, y = paired(a, b)
    lines = []
    for report, hits in ((a, x), (b, y)):
        lo, hi = wilson(sum(hits), len(hits))
        lines.append(f"{report['grounder']:40} {sum(hits) / len(hits):6.1%}  [{lo:.1%}, {hi:.1%}]  of {len(hits)}")
    only_a, only_b, p = mcnemar(x, y)
    lines.append(f"only the first right: {only_a}, only the second: {only_b}, McNemar exact p = {p:.2g}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in argv)
    print(compare(a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
