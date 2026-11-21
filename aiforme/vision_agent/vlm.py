"""A vision-language model as a grounder: the screenshot and the instruction in, a box to click out.

    VlmGrounder   Qwen2.5-VL served by Ollama, reached over HTTP

The model is asked for the element's box in the format Qwen2.5-VL was trained to ground in, `bbox_2d` in absolute
pixels of the image it was given, and the click is the box's centre. Qwen2.5-VL sees an image as 28×28 patches, so
the screenshot is resized here to a size it would choose itself — both sides multiples of 28, about a megapixel at
most — and its answer is scaled back. Resizing on this side means the coordinates never depend on how the server
would have resized the image, which differs between servers and versions.

That is not enough on its own: Ollama 0.40 still resizes some images itself (a 1904×364 window becomes about
2072×392 inside the server), and the model then answers in those pixels, not in the image it was sent. So before
the first command on each image size the grounder measures the server's frame: a dark image of that size with a red
square at a known place, and the ratio between where the model puts it and where it is. On the sizes where the
server keeps the image, that ratio is 1.

The server is at `AIFORME_VLM_URL` (default http://127.0.0.1:11434 — not `localhost`, which on Windows tries IPv6
first and pays a delay when the server listens on IPv4 only). Moving the model to another machine changes only that.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import urllib.request
from collections.abc import Callable

from aiforme.vision_agent.dataset import Snapshot

Point = tuple[int, int]
Post = Callable[[str, dict], dict]

PATCH = 28                       # Qwen2.5-VL's patch side in pixels
MAX_PIXELS = 1280 * PATCH * PATCH
MIN_PIXELS = 4 * PATCH * PATCH

_ASK = ("This is a screenshot of a window on a Windows PC. Find the UI element this command refers to: "
        "«{instruction}».\n"
        'Output its bounding box as JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "<its text>"}}].')
PROMPTS = {
    # v1 let the model decline. It declined far too often, even on a red square on a blank image at some sizes,
    # so v2 always asks for a box and counts a wrong click as a miss.
    "v1": _ASK + " If no such element is visible, output [].",
    "v2": _ASK,
}
PROBE = 'Find the red square. Output its bounding box as JSON: [{"bbox_2d": [x1, y1, x2, y2]}].'
PROBE_AT = 0.8                  # the probe square's centre, as a fraction of each side: far from the origin, so
                                # a scale error shows as a large offset


def model_size(width: int, height: int, max_pixels: int = MAX_PIXELS, min_pixels: int = MIN_PIXELS,
               factor: int = PATCH) -> tuple[int, int]:
    """The size Qwen2.5-VL's own preprocessing would pick (its `smart_resize`): both sides multiples of `factor`,
    the aspect ratio kept as closely as that allows, the area between the two limits."""
    w = max(factor, round(width / factor) * factor)
    h = max(factor, round(height / factor) * factor)
    if w * h > max_pixels:
        beta = math.sqrt(width * height / max_pixels)
        w = max(factor, math.floor(width / beta / factor) * factor)
        h = max(factor, math.floor(height / beta / factor) * factor)
    elif w * h < min_pixels:
        beta = math.sqrt(min_pixels / (width * height))
        w = math.ceil(width * beta / factor) * factor
        h = math.ceil(height * beta / factor) * factor
    return w, h


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    """The first `bbox_2d` in the model's answer. It usually answers with the JSON it was asked for, sometimes
    wrapped in a code fence or followed by prose, so the four numbers are found by pattern, not by parsing the
    whole reply. `[]` or no box at all → None."""
    m = re.search(r'"?bbox_2d"?\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]', text)
    if not m:
        return None
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def probe_image(size: tuple[int, int]) -> bytes:
    """A dark image of `size` with a red square centred at PROBE_AT of each side."""
    from PIL import Image, ImageDraw

    w, h = size
    picture = Image.new("RGB", size, (32, 32, 32))
    half = max(PATCH // 2, min(w, h) // 20)
    cx, cy = round(w * PROBE_AT), round(h * PROBE_AT)
    ImageDraw.Draw(picture).rectangle([cx - half, cy - half, cx + half, cy + half], fill=(230, 30, 30))
    buffer = io.BytesIO()
    picture.save(buffer, "PNG")
    return buffer.getvalue()


def http_post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(url, json.dumps(payload).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class VlmGrounder:
    """Asks the model where the element is. No answer, an empty answer or a box outside the image is None — a
    missed click. A server that cannot be reached raises: that is a broken setup, not a miss."""

    def __init__(self, model: str = "qwen2.5vl:7b", url: str | None = None, post: Post = http_post,
                 max_pixels: int = MAX_PIXELS, prompt: str = "v2", calibrate: bool = True):
        if prompt not in PROMPTS:
            raise ValueError(f"unknown prompt {prompt!r}: expected one of {tuple(PROMPTS)}")
        self.model = model
        self.url = (url or os.environ.get("AIFORME_VLM_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.post = post
        self.max_pixels = max_pixels
        self.prompt = prompt
        self.calibrate = calibrate
        self.name = f"vlm-{model.replace(':', '-').replace('/', '-')}-{prompt}" + ("" if calibrate else "-raw")
        self.last_reply = ""            # the model's raw answer, kept for the evaluation report
        self.frames: dict[tuple[int, int], tuple[float, float]] = {}   # image size → the server's scale, per axis

    def ask(self, content: str, png: bytes) -> str:
        reply = self.post(f"{self.url}/api/chat", {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 128},
            "messages": [{"role": "user", "content": content, "images": [base64.b64encode(png).decode("ascii")]}],
        })
        return reply.get("message", {}).get("content", "")

    def frame(self, size: tuple[int, int]) -> tuple[float, float]:
        """How many of the model's pixels there are per pixel of an image of `size`, measured once per size. If the
        model cannot find the probe, the frame stays at 1 and the server is trusted as before."""
        if not self.calibrate:
            return 1.0, 1.0
        if size not in self.frames:
            box = parse_box(self.ask(PROBE, probe_image(size)))
            scale = (1.0, 1.0)
            if box is not None:
                sx = (box[0] + box[2]) / 2 / (size[0] * PROBE_AT)
                sy = (box[1] + box[3]) / 2 / (size[1] * PROBE_AT)
                # within 2% is the model's own aim on one square, not a resized frame
                scale = (1.0 if abs(sx - 1) < 0.02 else sx, 1.0 if abs(sy - 1) < 0.02 else sy)
            self.frames[size] = scale
        return self.frames[size]

    def locate(self, snapshot: Snapshot, image: bytes | None, instruction: str) -> Point | None:
        self.last_reply = ""
        if image is None:
            return None
        from PIL import Image

        with Image.open(io.BytesIO(image)) as picture:
            width, height = picture.size
            size = model_size(width, height, self.max_pixels)
            buffer = io.BytesIO()
            picture.convert("RGB").resize(size, Image.Resampling.BICUBIC).save(buffer, "PNG")
        fx, fy = self.frame(size)
        self.last_reply = self.ask(PROMPTS[self.prompt].format(instruction=instruction), buffer.getvalue())
        box = parse_box(self.last_reply)
        if box is None:
            return None
        sx, sy = width / size[0] / fx, height / size[1] / fy
        x, y = round((box[0] + box[2]) / 2 * sx), round((box[1] + box[3]) / 2 * sy)
        return (x, y) if 0 <= x < width and 0 <= y < height else None
