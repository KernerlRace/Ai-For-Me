"""What a mode does to the machine, configured per mode in data/modes.json, and how it is undone.

    {"gaming":  {"notifications": false, "power_plan": "high", "close": []},
     "meeting": {"notifications": false},
     "coding":  {},
     "normal":  {}}

`notifications: false` turns Windows notification banners off; `power_plan: "high"` switches to High
performance; `close` lists apps to ask to close (gently, as if their window was closed). Whatever a mode
changes is remembered and put back when the machine leaves it, so «normal» is simply «as it was».
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

DEFAULTS = {
    "gaming": {"notifications": False, "power_plan": "high", "close": []},
    "meeting": {"notifications": False},
    "coding": {},
    "normal": {},
}

NOTIFICATIONS_KEY = r"Software\Microsoft\Windows\CurrentVersion\Notifications\Settings"
NOTIFICATIONS_VALUE = "NOC_GLOBAL_SETTING_TOASTS_ENABLED"
HIGH_PERFORMANCE = "8c5e7fda-e8bf-4a96-9a85-cf48a8f8a2a0"     # the built-in High performance scheme
CREATE_NO_WINDOW = 0x08000000


def load_settings(path: Path) -> dict:
    """The per-mode settings; written with the defaults on first use so they are easy to find and edit."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
        return {k: dict(v) for k, v in DEFAULTS.items()}
    settings = json.loads(path.read_text(encoding="utf-8"))
    return {mode: settings.get(mode, {}) for mode in DEFAULTS}


class System(Protocol):
    """The machine settings a mode can touch; WindowsSystem in use, a fake in tests."""
    def notifications(self) -> bool: ...
    def set_notifications(self, on: bool) -> None: ...
    def power_plan(self) -> str: ...
    def set_power_plan(self, guid: str) -> None: ...
    def close_app(self, name: str) -> None: ...


class ModeActions:
    def __init__(self, system: System, settings: dict):
        self.system, self.settings = system, settings
        self._saved: dict[str, object] = {}       # what was there before any mode changed it

    def switch(self, mode: str) -> list[str]:
        """Undoes the previous mode's changes, applies `mode`'s; returns what was done, for the log."""
        done = self.restore()
        cfg = self.settings.get(mode, {})
        try:
            if cfg.get("notifications") is False:
                self._save("notifications", self.system.notifications)
                self.system.set_notifications(False)
                done.append("notifications off")
            if cfg.get("power_plan") == "high":
                self._save("power_plan", self.system.power_plan)
                self.system.set_power_plan(HIGH_PERFORMANCE)
                done.append("power plan: high performance")
            for app in cfg.get("close", []):
                self.system.close_app(app)
                done.append(f"asked {app} to close")
        except Exception:
            log.exception("could not apply mode %s", mode)
        return done

    def restore(self) -> list[str]:
        done = []
        try:
            if "notifications" in self._saved:
                self.system.set_notifications(bool(self._saved.pop("notifications")))
                done.append("notifications restored")
            if "power_plan" in self._saved:
                self.system.set_power_plan(str(self._saved.pop("power_plan")))
                done.append("power plan restored")
        except Exception:
            log.exception("could not restore the settings")
        return done

    def _save(self, key: str, read) -> None:
        if key not in self._saved:
            self._saved[key] = read()


class WindowsSystem:
    def notifications(self) -> bool:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, NOTIFICATIONS_KEY) as key:
                return bool(winreg.QueryValueEx(key, NOTIFICATIONS_VALUE)[0])
        except FileNotFoundError:
            return True                              # never set: banners are on

    def set_notifications(self, on: bool) -> None:
        import winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, NOTIFICATIONS_KEY) as key:
            winreg.SetValueEx(key, NOTIFICATIONS_VALUE, 0, winreg.REG_DWORD, int(on))

    def power_plan(self) -> str:
        out = _run(["powercfg", "/getactivescheme"])
        match = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", out, re.IGNORECASE)
        if not match:
            raise RuntimeError(f"unexpected powercfg output: {out!r}")
        return match.group(0)

    def set_power_plan(self, guid: str) -> None:
        _run(["powercfg", "/setactive", guid])

    def close_app(self, name: str) -> None:
        image = name if name.lower().endswith(".exe") else f"{name}.exe"
        subprocess.run(["taskkill", "/im", image], capture_output=True, creationflags=CREATE_NO_WINDOW)


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout
