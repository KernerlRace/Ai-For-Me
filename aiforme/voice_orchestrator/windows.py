"""What the assistant can do on a Windows machine. Standard library only: ctypes for keys and the lock
screen, PowerShell for the list of installed apps, tasklist/taskkill for processes."""
from __future__ import annotations

import csv
import ctypes
import io
import json
import os
import subprocess
import time
import webbrowser
from ctypes import wintypes
from pathlib import Path
from urllib.parse import quote_plus

from aiforme.voice_orchestrator.apps import best_match, clean_name
from aiforme.voice_orchestrator.pipeline import ActionError

# virtual-key codes, https://learn.microsoft.com/windows/win32/inputdev/virtual-key-codes
VK = {"volume_mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF, "media_next": 0xB0,
      "media_play_pause": 0xB3, "lwin": 0x5B, "snapshot": 0x2C}

# apps that hide in the tray when their window is closed; only these are force-closed. Anything else
# may be holding unsaved work behind a "save changes?" dialog, so it is asked to close and left alone
TRAY_APPS = {"telegram", "discord", "spotify", "steam", "steamwebhelper"}
# never closed by voice
PROTECTED = {"explorer", "svchost", "csrss", "winlogon", "lsass", "dwm", "services", "system", "python",
             "pythonw", "powershell", "pwsh", "cmd", "conhost", "windowsterminal", "claude"}

SEARCH_DIRS = ("Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music", "OneDrive")
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "appdata", "$recycle.bin"}

CREATE_NO_WINDOW = 0x08000000


class WindowsDesktop:
    def __init__(self) -> None:
        self._apps: dict[str, str] | None = None

    # --- apps ---------------------------------------------------------------------------------------

    def apps(self) -> dict[str, str]:
        """Start menu name -> AppID, including Store apps; read once, a PowerShell call takes ~1 s."""
        if self._apps is None:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-StartApps | ConvertTo-Json -Compress"],
                capture_output=True, text=True, encoding="utf-8", creationflags=CREATE_NO_WINDOW, check=True,
            ).stdout
            rows = json.loads(out or "[]")
            rows = [rows] if isinstance(rows, dict) else rows
            self._apps = {r["Name"]: r["AppID"] for r in rows if r.get("Name") and r.get("AppID")}
        return self._apps

    def open_app(self, name: str) -> str:
        if normalize_simple(name) in {"browser", "браузер", "brauzer"}:
            webbrowser.open("https://www.google.com")
            return "browser"
        match = best_match(name, self.apps())
        if not match:
            raise ActionError(f"no installed app matches {name!r}", f"не нашлось приложения «{name}»")
        app = match[0]
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{self.apps()[app]}"])
        return app

    def close_app(self, name: str) -> str:
        running = _processes()
        candidates = sorted({p for p in running if clean_name(p) not in PROTECTED})
        # match against the Start menu name too: "word" finds WINWORD.EXE through "Word"
        match = best_match(name, candidates)
        if not match:
            app = best_match(name, self.apps())
            if app:
                match = best_match(app[0].split()[0], candidates)
        if not match:
            raise ActionError(f"no running app matches {name!r}", f"«{name}» не запущено")
        image = match[0]
        subprocess.run(["taskkill", "/im", image], capture_output=True, creationflags=CREATE_NO_WINDOW)
        if clean_name(image) in TRAY_APPS:
            time.sleep(2)
            if image in _processes():
                subprocess.run(["taskkill", "/im", image, "/f"], capture_output=True,
                               creationflags=CREATE_NO_WINDOW)
        return image

    # --- files and the web --------------------------------------------------------------------------

    def find_file(self, query: str, budget_s: float = 3.0) -> str:
        """Walks the usual user folders for up to `budget_s` seconds and shows the best match in Explorer."""
        home = Path(os.environ.get("USERPROFILE", Path.home()))
        names: dict[str, str] = {}
        deadline = time.monotonic() + budget_s
        for top in SEARCH_DIRS:
            for root, dirs, files in os.walk(home / top):
                dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
                for entry in (*dirs, *files):
                    names.setdefault(str(Path(root) / entry), Path(entry).stem)
                if time.monotonic() > deadline:
                    break
        match = best_match(query, names.values(), min_score=0.6)
        if not match:
            raise ActionError(f"nothing in your folders matches {query!r}",
                              f"в папках нет ничего похожего на «{query}»")
        path = next(p for p, stem in names.items() if stem == match[0])
        subprocess.Popen(["explorer.exe", f"/select,{path}"])
        return path

    def web_search(self, query: str) -> str:
        webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")
        return query

    # --- keyboard -----------------------------------------------------------------------------------

    def type_text(self, text: str) -> None:
        """Types into the focused window as Unicode key events, so Cyrillic works in any layout."""
        inputs = []
        for ch in text:
            for unit in _utf16_units(ch):
                inputs += [_key_input(scan=unit, flags=KEYEVENTF_UNICODE),
                           _key_input(scan=unit, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)]
        _send(inputs)

    def press(self, key: str, times: int = 1) -> None:
        _send([i for _ in range(times) for i in (_key_input(vk=VK[key]),
                                                  _key_input(vk=VK[key], flags=KEYEVENTF_KEYUP))])

    def screenshot(self) -> None:
        """Win+PrintScreen: Windows saves the screen to Pictures\\Screenshots."""
        _send([_key_input(vk=VK["lwin"]), _key_input(vk=VK["snapshot"]),
               _key_input(vk=VK["snapshot"], flags=KEYEVENTF_KEYUP),
               _key_input(vk=VK["lwin"], flags=KEYEVENTF_KEYUP)])

    def lock(self) -> None:
        if not ctypes.windll.user32.LockWorkStation():
            raise ActionError("Windows refused to lock the screen", "Windows не дала заблокировать экран")


def normalize_simple(text: str) -> str:
    return " ".join(text.lower().split())


def _processes() -> set[str]:
    out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True,
                         creationflags=CREATE_NO_WINDOW).stdout
    return {row[0] for row in csv.reader(io.StringIO(out)) if row}


# --- SendInput ------------------------------------------------------------------------------------------

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class MOUSEINPUT(ctypes.Structure):     # only here so the union has the size Windows expects
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _key_input(vk: int = 0, scan: int = 0, flags: int = 0) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD, u=_INPUTUNION(ki=KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags)))


def _utf16_units(ch: str) -> list[int]:
    data = ch.encode("utf-16-le")
    return [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]


def _send(inputs: list[INPUT]) -> None:
    if not inputs:
        return
    array = (INPUT * len(inputs))(*inputs)
    sent = ctypes.windll.user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise ActionError(f"Windows accepted {sent} of {len(inputs)} key events",
                          f"Windows приняла {sent} из {len(inputs)} нажатий")
