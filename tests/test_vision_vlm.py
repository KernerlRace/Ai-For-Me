import base64
import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from aiforme.vision_agent.dataset import Element, Snapshot, Store  # noqa: E402
from aiforme.vision_agent.evaluate import evaluate  # noqa: E402
from aiforme.vision_agent.grounding import make_grounder  # noqa: E402
from aiforme.vision_agent.vlm import PATCH, VlmGrounder, model_size, parse_box  # noqa: E402


def png(width, height) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, "PNG")
    return buffer.getvalue()


class FakeServer:
    """Answers like Ollama's /api/chat, with a box the test sets in the coordinates of the image it received."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        image = Image.open(io.BytesIO(base64.b64decode(payload["messages"][0]["images"][0])))
        text = self.answer(image.size) if callable(self.answer) else self.answer
        return {"message": {"role": "assistant", "content": text}}


def test_model_size_is_patch_aligned_and_bounded():
    for w, h in [(1920, 1080), (2560, 1440), (800, 600), (3840, 2160)]:
        mw, mh = model_size(w, h)
        assert mw % PATCH == 0 and mh % PATCH == 0
        assert mw * mh <= 1280 * PATCH * PATCH
        assert abs(mw / mh - w / h) / (w / h) < 0.05               # the aspect ratio survives
    # a thin strip loses its ratio to the patch grid; x and y are scaled back separately, so clicks still land
    assert model_size(300, 40) == (308, 28)
    assert model_size(812, 588) == (812, 588)                      # already aligned and small: left alone


def test_parse_box_reads_the_answer_however_it_is_wrapped():
    assert parse_box('[{"bbox_2d": [10, 20, 30, 40], "label": "Save"}]') == (10, 20, 30, 40)
    assert parse_box('```json\n[\n  {"bbox_2d": [30.5, 40, 10, 20]}\n]\n```') == (10, 20, 30.5, 40)
    assert parse_box("The button is here: bbox_2d: [1, 2, 3, 4] I think.") == (1, 2, 3, 4)
    assert parse_box("[]") is None
    assert parse_box("I cannot find it.") is None


def test_the_box_is_scaled_back_to_the_screenshot():
    # a 2560×1440 screenshot is sent smaller; the model answers in the small image's pixels
    def answer(size):
        w, h = size
        return f'[{{"bbox_2d": [{w * 0.25 - 10}, {h * 0.5 - 10}, {w * 0.25 + 10}, {h * 0.5 + 10}]}}]'

    server = FakeServer(answer)
    g = VlmGrounder("qwen2.5vl:7b", url="http://box:11434/", post=server)
    x, y = g.locate(Snapshot("s", 0, "", "x.exe", (2560, 1440)), png(2560, 1440), "сохранить")
    assert abs(x - 640) <= 2 and abs(y - 720) <= 2
    url, payload = server.calls[0]
    assert url == "http://box:11434/api/chat"
    assert payload["model"] == "qwen2.5vl:7b" and payload["options"]["temperature"] == 0
    assert "«сохранить»" in payload["messages"][0]["content"]
    assert g.last_reply.startswith('[{"bbox_2d"')


def test_no_box_or_a_box_off_the_image_is_no_click():
    snap = Snapshot("s", 0, "", "x.exe", (400, 300))
    assert VlmGrounder(post=FakeServer("[]")).locate(snap, png(400, 300), "шестерёнка") is None
    assert VlmGrounder(post=FakeServer('[{"bbox_2d": [900, 900, 950, 950]}]')).locate(
        snap, png(400, 300), "x") is None
    assert VlmGrounder(post=FakeServer("unused")).locate(snap, None, "x") is None


def test_the_address_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("AIFORME_VLM_URL", "http://inference-server:11434")
    assert VlmGrounder().url == "http://inference-server:11434"
    monkeypatch.delenv("AIFORME_VLM_URL")
    assert VlmGrounder().url == "http://127.0.0.1:11434"              # never localhost: IPv6 first on Windows
    assert make_grounder("vlm", "qwen2.5vl:3b").name == "vlm-qwen2.5vl-3b"


def test_evaluation_keeps_the_models_replies_and_can_sample(tmp_path):
    store = Store(tmp_path)
    elements = tuple(Element(f"Кнопка {i}", "Button", (i * 40, 0, i * 40 + 30, 20)) for i in range(10))
    store.add(Snapshot("s", 0, "", "x.exe", (400, 300), elements), png(400, 300))

    def answer(size):                         # always the first button, in the image the model was given
        sx, sy = size[0] / 400, size[1] / 300
        return f'[{{"bbox_2d": [{2 * sx}, {2 * sy}, {28 * sx}, {18 * sy}]}}]'

    result = evaluate(VlmGrounder(post=FakeServer(answer)), store, limit=5)
    assert result["n"] == 5
    assert [t["instruction"] for t in result["tasks"]] == ["Кнопка 0", "Кнопка 2", "Кнопка 4", "Кнопка 6", "Кнопка 8"]
    assert result["accuracy"] == pytest.approx(0.2)
    assert all(t["reply"].startswith("[{") for t in result["tasks"])
