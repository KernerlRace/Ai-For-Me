"""Grounding: a described element → the point on the screen to click.

    UiaGrounder       the baseline, no ML: the instruction matched against the names UI Automation reports
    VlmGrounder       a vision-language model reading the screenshot (vlm.py)
    SnapToElement     a grounder's click moved onto the nearest clickable element, when it landed on none
    FirstAnswer       the first grounder that answers: UI Automation where it knows the name, the model elsewhere

A grounder sees the snapshot — its image and its elements — and returns a point in the image's pixels, or None
when it finds nothing good enough; a wrong click costs more than "I couldn't find it".
"""
from __future__ import annotations

from typing import Protocol

from aiforme.vision_agent.dataset import CLICKABLE, Element, Snapshot
from aiforme.voice_orchestrator.nlu import normalize

Point = tuple[int, int]

# words a command puts before the name and the element never carries: «кнопку сохранить», "the save button"
FRAME_WORDS = frozenset([
    "кнопка", "кнопку", "кнопке", "вкладка", "вкладку", "пункт", "пункте", "ссылка", "ссылку", "поле", "меню",
    "галочка", "галочку", "флажок", "на", "в", "по",
    "button", "tab", "link", "item", "field", "box", "checkbox", "menu", "the", "on",
])


class Grounder(Protocol):
    name: str

    def locate(self, snapshot: Snapshot, image: bytes | None, instruction: str) -> Point | None: ...


def similarity(instruction: str, name: str) -> float:
    """The share of words in common (Jaccard), 1.0 when they match in any order. Frame words are dropped from the
    instruction only, and only if that helps: in «Закрыть вкладку» the word «вкладку» is the name itself, and
    dropping it there made the tab's close button as good a match for «закрыть» as the window's."""
    said, name_words = normalize(instruction).split(), set(normalize(name).split())
    if not name_words:
        return 0.0
    best = 0.0
    for a in (set(said), {w for w in said if w not in FRAME_WORDS}):
        if a:
            best = max(best, len(a & name_words) / len(a | name_words))
    return best


class UiaGrounder:
    """The element whose UI Automation name is closest to the instruction, if close enough. It cannot see the
    screen: an icon without a name, a canvas, a game or a web page that hides its tree is invisible to it."""
    name = "uia"

    def __init__(self, min_similarity: float = 0.5):
        self.min_similarity = min_similarity

    def locate(self, snapshot: Snapshot, image: bytes | None, instruction: str) -> Point | None:
        best: tuple[float, Element | None] = (0.0, None)
        for e in snapshot.elements:
            if e.type not in CLICKABLE or not e.enabled or not e.name:
                continue
            score = similarity(instruction, e.name)
            if score > best[0]:                 # ties keep the first in tree order: outer before nested
                best = (score, e)
        score, element = best
        return element.center if element is not None and score >= self.min_similarity else None


def _distance(point: Point, box: tuple[int, int, int, int]) -> float:
    """From a point to a box (right and bottom exclusive), 0 inside it."""
    x, y = point
    x0, y0, x1, y1 = box
    dx = max(x0 - x, 0, x - (x1 - 1))
    dy = max(y0 - y, 0, y - (y1 - 1))
    return (dx * dx + dy * dy) ** 0.5


class SnapToElement:
    """A model aims at what it sees, and a click a few pixels off a toggle or a row lands on nothing. When the
    window has an accessibility tree, a click that falls inside no clickable element is moved to the centre of the
    nearest one. A click that is already on an element is left alone, so this can only fix a near miss, never
    change a hit. With no tree — a game, a canvas — the click goes through as it is."""

    def __init__(self, inner: Grounder):
        self.inner = inner
        self.name = f"{inner.name}+snap"

    @property
    def last_reply(self) -> str:
        return getattr(self.inner, "last_reply", "")

    @property
    def frames(self) -> dict:
        return getattr(self.inner, "frames", {})

    def locate(self, snapshot: Snapshot, image: bytes | None, instruction: str) -> Point | None:
        point = self.inner.locate(snapshot, image, instruction)
        clickable = [e for e in snapshot.elements if e.type in CLICKABLE and e.enabled]
        if point is None or not clickable or any(_distance(point, e.box) == 0 for e in clickable):
            return point
        return min(clickable, key=lambda e: _distance(point, e.box)).center


class FirstAnswer:
    """The first grounder that gives an answer. UI Automation first: where an instruction is an element's name it
    is exact and costs nothing; where it is not, it declines and the model looks at the pixels."""

    def __init__(self, *grounders: Grounder):
        self.grounders = grounders
        self.name = "-then-".join(g.name for g in grounders)           # a file name too: no «|» on Windows
        self.last_reply = ""

    @property
    def frames(self) -> dict:
        return {k: v for g in self.grounders for k, v in getattr(g, "frames", {}).items()}

    def locate(self, snapshot: Snapshot, image: bytes | None, instruction: str) -> Point | None:
        self.last_reply = ""
        for g in self.grounders:
            point = g.locate(snapshot, image, instruction)
            self.last_reply = getattr(g, "last_reply", "") or f"({g.name})"
            if point is not None:
                return point
        return None


def make_grounder(name: str, model: str | None = None, prompt: str | None = None) -> Grounder:
    """uia · vlm · vlm-snap (the model, its click snapped to an element) · hybrid (uia, then vlm-snap)."""
    if name == "uia":
        return UiaGrounder()
    if name in ("vlm", "vlm-snap", "hybrid"):
        from aiforme.vision_agent.vlm import VlmGrounder
        options = {"model": model, "prompt": prompt}
        vlm = VlmGrounder(**{k: v for k, v in options.items() if v})
        if name == "vlm":
            return vlm
        if name == "vlm-snap":
            return SnapToElement(vlm)
        return FirstAnswer(UiaGrounder(), SnapToElement(vlm))
    raise ValueError(f"unknown grounder {name!r}: expected one of {GROUNDERS}")


GROUNDERS = ("uia", "vlm", "vlm-snap", "hybrid")
