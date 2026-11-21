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
    assert summarize([]) == {"n": 0, "accuracy": 0.0, "no_answer": 0.0, "wrong_click": 0.0, "median_seconds": None,
                             "by_app": {}, "by_type": {}}


def test_described_tasks_point_at_elements_and_fail_loudly(tmp_path):
    from aiforme.vision_agent.dataset import described_tasks

    path = tmp_path / "described.tsv"
    path.write_text("# snapshot\telement\tinstruction\n\ns1\t4\tгалочка тёмного режима\ns1\t0\tдискетка\n",
                    encoding="utf-8")
    assert described_tasks([SNAP], path) == [
        Task("s1", "галочка тёмного режима", (10, 100, 120, 120), "CheckBox", "Settings.exe"),
        Task("s1", "дискетка", (10, 10, 90, 40), "Button", "Settings.exe"),
    ]
    path.write_text("s1\t99\tнет такого\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no element 99"):
        described_tasks([SNAP], path)


def test_evaluate_on_described_tasks(tmp_path):
    store = Store(tmp_path)
    store.add(SNAP, b"")
    store.described.write_text("s1\t0\tдискетка\n", encoding="utf-8")
    result = evaluate(UiaGrounder(), store, task_set="described")
    assert (result["task_set"], result["n"], result["accuracy"], result["no_answer"]) == ("described", 1, 0.0, 1.0)
    with pytest.raises(ValueError):
        evaluate(UiaGrounder(), store, task_set="nonsense")


def test_a_near_miss_is_snapped_onto_the_nearest_element_and_a_hit_is_left_alone():
    from aiforme.vision_agent.grounding import SnapToElement

    class Fixed:
        name = "fixed"

        def __init__(self, point):
            self.point = point

        def locate(self, snapshot, image, instruction):
            return self.point

    toggle = Element("Wi-Fi", "Button", (100, 100, 140, 120))
    row = Element("Сеть", "ListItem", (0, 200, 300, 240))
    label = Element("Беспроводная сеть", "Text", (0, 100, 90, 120))       # not clickable: never a target
    snap = Snapshot("s", 0, "", "x.exe", (400, 300), (toggle, row, label))
    assert SnapToElement(Fixed((95, 110))).locate(snap, None, "") == toggle.center      # 5 px left of the toggle
    assert SnapToElement(Fixed((150, 230))).locate(snap, None, "") == (150, 230)        # inside the row: as is
    assert SnapToElement(Fixed(None)).locate(snap, None, "") is None
    bare = Snapshot("s", 0, "", "game.exe", (400, 300))                                 # no tree: as is
    assert SnapToElement(Fixed((95, 110))).locate(bare, None, "") == (95, 110)
    assert SnapToElement(Fixed(None)).name == "fixed+snap"


def test_the_hybrid_asks_ui_automation_first_and_the_model_only_when_it_declines():
    from aiforme.vision_agent.grounding import FirstAnswer, UiaGrounder

    class Model:
        name = "model"
        last_reply = "[]"

        def __init__(self):
            self.asked = 0

        def locate(self, snapshot, image, instruction):
            self.asked += 1
            return (5, 5)

    save = Element("Сохранить", "Button", (10, 10, 50, 30))
    snap = Snapshot("s", 0, "", "x.exe", (400, 300), (save,))
    model = Model()
    hybrid = FirstAnswer(UiaGrounder(), model)
    assert hybrid.locate(snap, None, "сохранить") == save.center and model.asked == 0
    assert hybrid.locate(snap, None, "шестерёнка") == (5, 5) and model.asked == 1
    assert hybrid.name == "uia-then-model"


def test_wilson_and_mcnemar():
    from aiforme.vision_agent.compare import mcnemar, paired, wilson

    lo, hi = wilson(33, 59)
    assert lo == pytest.approx(0.433, abs=0.002) and hi == pytest.approx(0.678, abs=0.002)
    assert wilson(10, 10)[1] == 1.0 and wilson(0, 10)[0] == 0.0
    assert mcnemar([False] * 13 + [True] * 46, [True] * 59) == (0, 13, pytest.approx(2 / 2 ** 13))
    assert mcnemar([True, False], [True, False]) == (0, 0, 1.0)
    a = {"tasks": [{"snapshot": "s", "instruction": i, "hit": h} for i, h in [("x", True), ("y", False)]]}
    b = {"tasks": [{"snapshot": "s", "instruction": i, "hit": h} for i, h in [("y", True), ("z", True)]]}
    assert paired(a, b) == ([False], [True])
