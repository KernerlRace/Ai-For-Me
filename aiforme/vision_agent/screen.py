"""The active window through Windows' UI Automation: its image and every element in it. Windows only.

The process is made per-monitor DPI aware first, so UI Automation's rectangles, the window's bounds and the screen
capture all count physical pixels — at 125% scaling they would otherwise disagree by a quarter.
"""
from __future__ import annotations

import ctypes
import io
import time
from collections.abc import Iterator
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

from aiforme.vision_agent.dataset import Element, Snapshot, clip

# UI Automation property and control type ids (UIAutomationClient.h)
NAME, CONTROL_TYPE, RECT, ENABLED = 30005, 30003, 30001, 30010
OFFSCREEN, AUTOMATION_ID, CLASS_NAME = 30022, 30011, 30012
TREE_SCOPE_DESCENDANTS = 4
CONTROL_TYPES = {
    50000: "Button", 50001: "Calendar", 50002: "CheckBox", 50003: "ComboBox", 50004: "Edit", 50005: "Hyperlink",
    50006: "Image", 50007: "ListItem", 50008: "List", 50009: "Menu", 50010: "MenuBar", 50011: "MenuItem",
    50012: "ProgressBar", 50013: "RadioButton", 50014: "ScrollBar", 50015: "Slider", 50016: "Spinner",
    50017: "StatusBar", 50018: "Tab", 50019: "TabItem", 50020: "Text", 50021: "ToolBar", 50022: "ToolTip",
    50023: "Tree", 50024: "TreeItem", 50025: "Custom", 50026: "Group", 50027: "Thumb", 50028: "DataGrid",
    50029: "DataItem", 50030: "Document", 50031: "SplitButton", 50032: "Window", 50033: "Pane", 50034: "Header",
    50035: "HeaderItem", 50036: "Table", 50037: "TitleBar", 50038: "Separator", 50039: "SemanticZoom",
    50040: "AppBar",
}
MAX_ELEMENTS = 5000         # a browser page can report tens of thousands; the first ones in tree order are kept

_user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None


def dpi_aware() -> None:
    """Per-monitor DPI awareness v2 for this process; once set, later calls fail harmlessly."""
    _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))


def foreground() -> int:
    return _user32.GetForegroundWindow()


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """The window's visible bounds: without the invisible resize border GetWindowRect includes on Windows 10+."""
    rect = wintypes.RECT()
    dwm = ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect))   # 9: frame
    if dwm != 0:
        _user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(512)
    _user32.GetWindowTextW(hwnd, buf, len(buf))
    return buf.value


def window_process(hwnd: int) -> str:
    import psutil
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    try:
        return psutil.Process(pid.value).name()
    except psutil.Error:
        return ""


class Automation:
    """One UI Automation client; reading a whole window costs a single cross-process call thanks to the cache."""

    def __init__(self) -> None:
        import comtypes.client
        comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient as uia
        self._uia = comtypes.client.CreateObject(uia.CUIAutomation, interface=uia.IUIAutomation)
        self._cache = self._uia.CreateCacheRequest()
        for prop in (NAME, CONTROL_TYPE, RECT, ENABLED, OFFSCREEN, AUTOMATION_ID, CLASS_NAME):
            self._cache.AddProperty(prop)

    def elements(self, hwnd: int, origin: tuple[int, int], size: tuple[int, int]) -> list[Element]:
        """Every visible element of the window, its box moved into the window image's pixels and clipped to it."""
        root = self._uia.ElementFromHandle(hwnd)
        found = root.FindAllBuildCache(TREE_SCOPE_DESCENDANTS, self._uia.CreateTrueCondition(), self._cache)
        ox, oy = origin
        out = []
        for i in range(min(found.Length, MAX_ELEMENTS)):
            e = found.GetElement(i)
            if e.CachedIsOffscreen:
                continue
            r = e.CachedBoundingRectangle
            visible = clip((r.left - ox, r.top - oy, r.right - ox, r.bottom - oy), size)
            if visible is None:
                continue
            box, clipped = visible
            out.append(Element(
                name=(e.CachedName or "").strip(), type=CONTROL_TYPES.get(e.CachedControlType, "Unknown"), box=box,
                enabled=bool(e.CachedIsEnabled), clipped=clipped, automation_id=e.CachedAutomationId or "",
                class_name=e.CachedClassName or ""))
        return out


def capture(rect: tuple[int, int, int, int]) -> bytes:
    """A PNG of the screen inside `rect` (screen coordinates; every monitor counts)."""
    from PIL import ImageGrab
    image = ImageGrab.grab(bbox=rect, all_screens=True)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def snapshot(automation: Automation, hwnd: int | None = None) -> tuple[Snapshot, bytes]:
    """The window (the foreground one by default) as it is now: its image and its elements."""
    hwnd = hwnd or foreground()
    left, top, right, bottom = window_rect(hwnd)
    size = (right - left, bottom - top)
    elements = automation.elements(hwnd, (left, top), size)
    png = capture((left, top, right, bottom))
    now = time.time()
    snap_id = datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S-%f")[:-3]
    snap = Snapshot(snap_id, now, window_title(hwnd), window_process(hwnd), size, tuple(elements))
    return snap, png


def collect(root: str | Path, count: int = 1, delay: float = 3.0, interval: float = 10.0) -> Iterator[Snapshot]:
    """`count` snapshots of whatever window is in front, saved as they are taken, the first after `delay`
    seconds — time to switch to the window."""
    from aiforme.vision_agent.dataset import Store
    dpi_aware()
    store, automation = Store(root), Automation()
    time.sleep(delay)
    for i in range(count):
        if i:
            time.sleep(interval)
        snap, png = snapshot(automation)
        store.add(snap, png)
        yield snap
