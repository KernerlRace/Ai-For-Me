"""A vision-language model as a grounder: the screenshot and the instruction in, a box to click out.

    VlmGrounder   Qwen2.5-VL served by Ollama, reached over HTTP

The model is asked for the element's box in the format Qwen2.5-VL was trained to ground in, `bbox_2d` in absolute
pixels of the image it was given, and the click is the box's centre. Qwen2.5-VL sees an image as 28×28 patches, so
the screenshot is resized here to a size it would choose itself — both sides multiples of 28, about a megapixel at
most — and its answer is scaled back. Resizing on this side means the coordinates never depend on how the server
would have resized the image, which differs between servers and versions.

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

PROMPT = ("This is a screenshot of a window on a Windows PC. Find the UI element this command refers to: "
          "«{instruction}».\n"
          'Output its bounding box as JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "<its text>"}}]. '
          "If no such element is visible, output [].")


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


def http_post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(url, json.dumps(payload).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class VlmGrounder:
    """Asks the model where the element is. No answer, an empty answer or a box outside the image is None — a
    missed click, never a guess. A server that cannot be reached raises: that is a broken setup, not a miss."""

    def __init__(self, model: str = "qwen2.5vl:7b", url: str | None = None, post: Post = http_post,
                 max_pixels: int = MAX_PIXELS):
        self.model = model
        self.url = (url or os.environ.get("AIFORME_VLM_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.post = post
        self.max_pixels = max_pixels
        self.name = f"vlm-{model.replace(':', '-').replace('/', '-')}"
        self.last_reply = ""            # the model's raw answer, kept for the evaluation report

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
        reply = self.post(f"{self.url}/api/chat", {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 128},
            "messages": [{"role": "user", "content": PROMPT.format(instruction=instruction),
                          "images": [base64.b64encode(buffer.getvalue()).decode("ascii")]}],
        })
        self.last_reply = reply.get("message", {}).get("content", "")
        box = parse_box(self.last_reply)
        if box is None:
            return None
        sx, sy = width / size[0], height / size[1]
        x, y = round((box[0] + box[2]) / 2 * sx), round((box[1] + box[3]) / 2 * sy)
        return (x, y) if 0 <= x < width and 0 <= y < height else None
