"""What the machine is doing right now: the app in front, and which kinds of apps are running.

An app is put into a category by its executable: games by where they are installed (any Steam, Epic, Riot,
Wargaming, GOG or Xbox library folder) or by a short list of names, IDEs and terminals by name, and so on.
Launchers are not games — Steam open in the tray is not playing.
"""
from __future__ import annotations

import ctypes
import time
from collections.abc import Iterable
from ctypes import wintypes
from dataclasses import dataclass, field

CATEGORIES = ("game", "ide", "terminal", "meeting", "browser", "other")

GAME_FOLDERS = ("\\steamapps\\common\\", "\\epic games\\", "\\riot games\\", "\\wargaming.net\\games\\",
                "\\gog galaxy\\games\\", "\\xboxgames\\", "\\battle.net\\games\\")
LAUNCHERS = {"steam", "steamwebhelper", "epicgameslauncher", "riotclientservices", "riotclientux", "wgc",
             "battle.net", "galaxyclient", "gamebar", "steamservice", "gameoverlayui"}
GAMES = {"dota2", "cs2", "worldoftanks", "wotblitz", "valorant-win64-shipping", "league of legends", "r5apex",
         "fortniteclient-win64-shipping", "robloxplayerbeta", "gta5", "rocketleague", "overwatch", "eldenring",
         "minecraft.windows", "cyberpunk2077", "witcher3", "baldursgate3", "bg3", "destiny2", "pubg"}
IDES = {"code", "cursor", "windsurf", "pycharm64", "idea64", "rider64", "webstorm64", "clion64", "goland64",
        "devenv", "sublime_text", "zed", "notepad++", "android studio", "studio64"}
TERMINALS = {"windowsterminal", "wt", "powershell", "pwsh", "cmd", "conhost", "alacritty", "wezterm-gui"}
MEETING_APPS = {"zoom", "ms-teams", "teams", "webex", "ciscocollabhost", "skype"}
BROWSERS = {"chrome", "msedge", "firefox", "opera", "brave", "vivaldi", "yandex", "arc"}


def categorize(name: str, exe: str = "") -> str:
    """'Code.exe' -> 'ide'; anything under steamapps\\common that is not a launcher -> 'game'."""
    stem = name.lower().removesuffix(".exe")
    path = exe.lower()
    if stem in LAUNCHERS:
        return "other"
    if stem in GAMES or any(folder in path for folder in GAME_FOLDERS):
        return "game"
    for category, names in (("ide", IDES), ("terminal", TERMINALS), ("meeting", MEETING_APPS),
                            ("browser", BROWSERS)):
        if stem in names:
            return category
    return "other"


@dataclass(frozen=True)
class Activity:
    time: float
    foreground: str = ""                       # process name of the active window, e.g. "Code.exe"
    title: str = ""                            # its window title
    category: str = "other"                    # category of the foreground app
    running: frozenset[str] = field(default_factory=frozenset)      # categories of everything running
    games: tuple[str, ...] = ()                # running games, for the reason given when the mode changes


class WindowsActivity:
    """Samples the foreground window with the Win32 API and the process list with psutil."""

    def sample(self) -> Activity:
        import psutil
        name, exe, title = _foreground()
        running, games = set(), []
        for proc in psutil.process_iter(["name", "exe"]):
            info = proc.info
            category = categorize(info.get("name") or "", info.get("exe") or "")
            running.add(category)
            if category == "game":
                games.append(info["name"])
        return Activity(time.time(), name, title, categorize(name, exe), frozenset(running), tuple(sorted(set(games))))


def _foreground() -> tuple[str, str, str]:
    """(process name, executable path, window title) of the active window; empty strings if unknown."""
    import psutil
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return "", "", ""
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    try:
        proc = psutil.Process(pid.value)
        return proc.name(), proc.exe(), buf.value
    except (psutil.Error, OSError):
        return "", "", buf.value


def share(samples: Iterable[Activity], categories: set[str]) -> float:
    """Fraction of the samples whose foreground app is in one of `categories`."""
    samples = list(samples)
    return sum(s.category in categories for s in samples) / len(samples) if samples else 0.0
