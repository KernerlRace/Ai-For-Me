import json

import pytest

from aiforme.vision_agent.dataset import Element, Snapshot, Store, Task, clip, targets, tasks
from aiforme.vision_agent.evaluate import evaluate, summarize
from aiforme.vision_agent.grounding import UiaGrounder, similarity


def el(name, type="Button", box=(10, 10, 50, 30), **kw):
    return Element(name, type, box, **kw)


SNAP = Snapshot("s1", 1.0, "Настройки", "Settings.exe", (400, 300), (
    el("Сохранить", box=(10, 10, 90, 40)),
    el("Отмена", box=(100, 10, 180, 40)),
    el("Закрыть", box=(370, 0, 400, 20)),
    el("Закрыть", type="TabItem", box=(10, 60, 90, 80)),           # a second «Закрыть»: ambiguous
    el("Тёмная тема", type="CheckBox", box=(10, 100, 120, 120)),
    el("Громкость", type="Slider", box=(10, 140, 200, 160)),       # not something you click by name
    el("Экспорт", box=(10, 180, 90, 200), enabled=False),
    el("Обновить", box=(380, 280, 400, 300), clipped=True),
    el("", box=(200, 200, 230, 230)),                              # an icon without a name
))


def test_clip_keeps_the_visible_part():
    assert clip((10, 10, 50, 30), (400, 300)) == ((10, 10, 50, 30), False)
    assert clip((-20, 10, 50, 30), (400, 300)) == ((0, 10, 50, 30), True)
    assert clip((398, 10, 450, 30), (400, 300)) is None                   # 2 px left: too thin
    assert clip((500, 500, 600, 600), (400, 300)) is None


def test_targets_are_clickable_unambiguous_and_in_view():
    assert [e.name for e in targets(SNAP)] == ["Сохранить", "Отмена", "Тёмная тема"]
    task = tasks([SNAP])[0]
    assert task == Task("s1", "Сохранить", (10, 10, 90, 40), "Button", "Settings.exe")
    assert task.hit((10, 10)) and task.hit((89, 39)) and not task.hit((90, 20)) and not task.hit(None)


def test_snapshots_round_trip_through_the_store(tmp_path):
    store = Store(tmp_path / "vision")
    store.add(SNAP, b"png")
    store.add(Snapshot("s2", 2.0, "", "x.exe", (10, 10)), b"png2")
    assert store.snapshots() == [SNAP, Snapshot("s2", 2.0, "", "x.exe", (10, 10))]
    assert store.image("s2").read_bytes() == b"png2"
    row = json.loads(store.index.read_text(encoding="utf-8").splitlines()[0])
    assert row["elements"][4] == {"name": "Тёмная тема", "type": "CheckBox", "box": [10, 100, 120, 120],
                                  "enabled": True, "clipped": False, "automation_id": "", "class_name": ""}


def test_similarity_ignores_case_order_and_frame_words():
    assert similarity("кнопку Сохранить", "Сохранить") == 1.0
    assert similarity("the Save button", "save") == 1.0
    assert similarity("тема тёмная", "Тёмная тема") == 1.0
    assert similarity("тёмная", "Тёмная тема") == pytest.approx(0.5)
    assert similarity("кнопку", "Сохранить") == 0.0
    # a frame word inside a name is part of the name: the window's «Закрыть» beats the tab's «Закрыть вкладку»
    assert similarity("закрыть", "Закрыть вкладку") == pytest.approx(0.5)
    assert similarity("закрыть вкладку", "Закрыть вкладку") == 1.0


def test_the_exact_name_wins_over_one_with_a_frame_word():
    snap = Snapshot("s", 0.0, "", "Notepad.exe", (500, 100),
                    (el("Закрыть вкладку", box=(10, 10, 40, 30)), el("Закрыть", box=(450, 0, 500, 30))))
    assert UiaGrounder().locate(snap, None, "Закрыть") == (475, 15)
    assert UiaGrounder().locate(snap, None, "закрыть вкладку") == (25, 20)


def test_uia_grounder_clicks_the_closest_name_or_nothing():
    g = UiaGrounder()
    assert g.locate(SNAP, None, "сохранить") == (50, 25)
    assert g.locate(SNAP, None, "галочку тёмная тема") == (65, 110)
    assert g.locate(SNAP, None, "громкость") is None           # a slider: not clickable by name
    assert g.locate(SNAP, None, "экспорт") is None             # disabled
    assert g.locate(SNAP, None, "шестерёнка") is None          # nothing like it: better no click than a wrong one


def test_evaluate_scores_hits_by_app_and_type(tmp_path):
    store = Store(tmp_path)
    store.add(SNAP, b"")

    class Off:
        name = "off"

        def locate(self, snapshot, image, instruction):
            return None if instruction == "Отмена" else (0, 0)

    assert evaluate(UiaGrounder(), store)["accuracy"] == 1.0
    result = evaluate(Off(), store)
    assert (result["n"], result["accuracy"], result["no_answer"]) == (3, 0.0, pytest.approx(1 / 3))
    assert result["by_type"]["CheckBox"]["n"] == 1 and result["by_app"]["Settings.exe"]["n"] == 3


def test_summarize_handles_no_tasks():
    assert summarize([]) == {"n": 0, "accuracy": 0.0, "no_answer": 0.0, "median_seconds": None,
                             "by_app": {}, "by_type": {}}
