import base64
import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from aiforme.vision_agent.dataset import Element, Snapshot, Store  # noqa: E402
from aiforme.vision_agent.evaluate import evaluate  # noqa: E402
from aiforme.vision_agent.grounding import make_grounder  # noqa: E402
from aiforme.vision_agent.vlm import PATCH, PROBE, VlmGrounder, model_size, parse_box  # noqa: E402


def png(width, height) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, "PNG")
    return buffer.getvalue()


class FakeServer:
    """Answers like Ollama's /api/chat, with a box the test sets in the coordinates of the image it received.

    `zoom` plays a server that resizes the image itself, as Ollama does with some sizes: the model then sees, and
    answers in, an image `zoom` times larger than the one it was sent. The red probe square is found by its pixels."""

    def __init__(self, answer, zoom=(1.0, 1.0)):
        self.answer = answer
        self.zoom = zoom
        self.calls = []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        message = payload["messages"][0]
        image = Image.open(io.BytesIO(base64.b64decode(message["images"][0]))).convert("RGB")
        if message["content"] == PROBE:
            red = [(x, y) for x in range(0, image.width, 2) for y in range(0, image.height, 2)
                   if image.getpixel((x, y))[0] > 200]
            box = (min(x for x, _ in red), min(y for _, y in red), max(x for x, _ in red), max(y for _, y in red))
            text = f'[{{"bbox_2d": {list(box)}}}]'
        else:
            text = self.answer(image.size) if callable(self.answer) else self.answer
        zx, zy = self.zoom
        return {"message": {"role": "assistant", "content": _zoom(text, zx, zy)}}

    def asked(self):
        """The calls that were commands, not probes."""
        return [(u, p) for u, p in self.calls if p["messages"][0]["content"] != PROBE]


def _zoom(text, zx, zy):
    box = parse_box(text)
    if box is None:
        return text
    x0, y0, x1, y1 = box
    return f'[{{"bbox_2d": [{x0 * zx:.0f}, {y0 * zy:.0f}, {x1 * zx:.0f}, {y1 * zy:.0f}]}}]'


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
    url, payload = server.asked()[0]
    assert url == "http://box:11434/api/chat"
    assert payload["model"] == "qwen2.5vl:7b" and payload["options"]["temperature"] == 0
    assert "«сохранить»" in payload["messages"][0]["content"]
    assert g.last_reply.startswith('[{"bbox_2d"')
    assert g.frames == {model_size(2560, 1440): (1.0, 1.0)}                 # a server that keeps the image: frame 1


def test_a_server_that_resizes_the_image_is_measured_once_per_size():
    # Ollama turns a 1904×364 image into about 2072×392 and the model answers in those pixels
    def answer(size):                       # the element is the box (1500, 100)–(1540, 140) of the image sent
        return '[{"bbox_2d": [1500, 100, 1540, 140]}]'

    server = FakeServer(answer, zoom=(2072 / 1904, 392 / 364))
    g = VlmGrounder(post=server)
    snap = Snapshot("s", 0, "", "explorer.exe", (1906, 362))
    for _ in range(3):
        x, y = g.locate(snap, png(1906, 362), "обновить")
        assert abs(x - 1520) <= 3 and abs(y - 120) <= 3                # where it is, not 9% to the right
    assert len(server.calls) == 4                                    # one probe, three commands
    assert g.frames[(1904, 364)][0] == pytest.approx(2072 / 1904, abs=0.01)

    raw = VlmGrounder(post=FakeServer(answer, zoom=(2072 / 1904, 392 / 364)), calibrate=False)
    x, _ = raw.locate(snap, png(1906, 362), "обновить")
    assert x > 1600                                                  # trusting the server's frame misses
    assert raw.name == "vlm-qwen2.5vl-7b-v2-raw"


def test_the_v1_prompt_lets_the_model_decline_and_v2_does_not():
    v1, v2 = FakeServer("[]"), FakeServer("[]")
    snap = Snapshot("s", 0, "", "x.exe", (400, 300))
    VlmGrounder(prompt="v1", post=v1).locate(snap, png(400, 300), "x")
    VlmGrounder(prompt="v2", post=v2).locate(snap, png(400, 300), "x")
    assert "output []" in v1.asked()[0][1]["messages"][0]["content"]
    assert "output []" not in v2.asked()[0][1]["messages"][0]["content"]
    with pytest.raises(ValueError):
        VlmGrounder(prompt="v9")


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
    assert make_grounder("vlm", "qwen2.5vl:3b").name == "vlm-qwen2.5vl-3b-v2"
    assert make_grounder("vlm", None, "v1").name == "vlm-qwen2.5vl-7b-v1"


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
    assert result["wrong_click"] == pytest.approx(0.8)               # always the first button: 4 of 5 land wrong
    w, h = model_size(400, 300)
    assert result["frames"] == {f"{w}x{h}": [1.0, 1.0]}
    assert all(t["reply"].startswith("[{") for t in result["tasks"])
