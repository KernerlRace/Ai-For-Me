"""Grounding: a described element → the point on the screen to click.

    UiaGrounder   the baseline, no ML: the instruction matched against the names UI Automation reports
    VlmGrounder   a vision-language model reading the screenshot (vlm.py)

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


def make_grounder(name: str, model: str | None = None) -> Grounder:
    if name == "uia":
        return UiaGrounder()
    if name == "vlm":
        from aiforme.vision_agent.vlm import VlmGrounder
        return VlmGrounder(model) if model else VlmGrounder()
    raise ValueError(f"unknown grounder {name!r}: expected one of {GROUNDERS}")


GROUNDERS = ("uia", "vlm")
