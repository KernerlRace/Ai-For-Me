"""The grounding dataset: snapshots of real windows, and the click targets derived from them.

A snapshot is one window as it looked: a PNG of it and every UI element Windows' UI Automation reported in it, with
its name, control type and box in the image's pixels. UI Automation knows exactly where each element is, so the
boxes are ground truth that costs no hand labelling.

A task is one element a command could name: «Сохранить» → the box of the Save button. The instruction is the
element's own name, so these are literal tasks — what a VLM must do is read the screen and find the words; the
UI Automation baseline, which reads the same names, is all but an upper bound on them. Commands in my own words
(«нажми на шестерёнку») come later, written by hand against the same snapshots.

    data/vision/snapshots.jsonl     one snapshot per line
    data/vision/<snapshot id>.png   its image
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aiforme.voice_orchestrator.nlu import normalize

Box = tuple[int, int, int, int]          # left, top, right, bottom in image pixels; right and bottom exclusive

# what a person clicks; Text, Pane, Group and the like are layout, rarely a target of their own
CLICKABLE = frozenset([
    "Button", "CheckBox", "ComboBox", "DataItem", "Edit", "HeaderItem", "Hyperlink", "ListItem", "MenuItem",
    "RadioButton", "SplitButton", "TabItem", "TreeItem",
])
MAX_NAME = 60           # longer names are whole messages or documents, not something said in a command
MIN_SIDE = 4            # pixels: smaller boxes are separators and hidden stubs


@dataclass(frozen=True)
class Element:
    name: str
    type: str                   # UI Automation control type without the suffix: "Button", "ListItem"
    box: Box
    enabled: bool = True
    clipped: bool = False       # partly outside the window: only the visible part is in `box`
    automation_id: str = ""
    class_name: str = ""

    @property
    def center(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) // 2, (y0 + y1) // 2


@dataclass(frozen=True)
class Snapshot:
    id: str
    taken_at: float
    window: str                 # title
    process: str                # executable name: "Telegram.exe"
    size: tuple[int, int]       # image width, height
    elements: tuple[Element, ...] = field(default_factory=tuple)

    def to_json(self) -> dict:
        return {**asdict(self), "elements": [asdict(e) for e in self.elements]}

    @classmethod
    def from_json(cls, row: dict) -> Snapshot:
        elements = tuple(Element(**{**e, "box": tuple(e["box"])}) for e in row["elements"])
        return cls(row["id"], row["taken_at"], row["window"], row["process"], tuple(row["size"]), elements)


@dataclass(frozen=True)
class Task:
    snapshot: str
    instruction: str
    box: Box
    type: str
    app: str

    def hit(self, point: tuple[int, int] | None) -> bool:
        if point is None:
            return False
        x, y = point
        x0, y0, x1, y1 = self.box
        return x0 <= x < x1 and y0 <= y < y1


def clip(box: Box, size: tuple[int, int]) -> tuple[Box, bool] | None:
    """The part of `box` inside an image of `size`, and whether anything was cut; None if too little is left."""
    w, h = size
    x0, y0, x1, y1 = box
    cut = (max(x0, 0), max(y0, 0), min(x1, w), min(y1, h))
    if cut[2] - cut[0] < MIN_SIDE or cut[3] - cut[1] < MIN_SIDE:
        return None
    return cut, cut != tuple(box)


def targets(snapshot: Snapshot) -> list[Element]:
    """Elements a literal command can name without ambiguity: clickable, enabled, wholly in view, with a short
    name no other such element in the window shares (two «Закрыть» buttons — the instruction cannot say which)."""
    candidates = [e for e in snapshot.elements
                  if e.type in CLICKABLE and e.enabled and not e.clipped and 0 < len(e.name.strip()) <= MAX_NAME
                  and normalize(e.name)]
    counts = Counter(normalize(e.name) for e in candidates)
    return [e for e in candidates if counts[normalize(e.name)] == 1]


def tasks(snapshots: Iterable[Snapshot]) -> list[Task]:
    return [Task(s.id, e.name.strip(), e.box, e.type, s.process) for s in snapshots for e in targets(s)]


class Store:
    """The snapshots on disk: appended one line at a time, so a crash mid-collection loses one at most."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index = self.root / "snapshots.jsonl"

    def image(self, snapshot_id: str) -> Path:
        return self.root / f"{snapshot_id}.png"

    def add(self, snapshot: Snapshot, png: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.image(snapshot.id).write_bytes(png)
        with self.index.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot.to_json(), ensure_ascii=False) + "\n")

    def snapshots(self) -> list[Snapshot]:
        if not self.index.exists():
            return []
        return [Snapshot.from_json(json.loads(line))
                for line in self.index.read_text(encoding="utf-8").splitlines() if line.strip()]
