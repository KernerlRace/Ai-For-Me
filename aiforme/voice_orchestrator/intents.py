"""The intents the voice orchestrator understands in Phase 1, and the slot each one takes."""
from __future__ import annotations

OUT_OF_SCOPE = "out_of_scope"   # anything that is not one of the commands below

# intent -> its slot, or None
INTENTS: dict[str, str | None] = {
    "open_app": "app",
    "close_app": "app",
    "find_file": "query",
    "dictate": "text",
    "web_search": "query",
    "volume_up": None,
    "volume_down": None,
    "mute": None,
    "media_play_pause": None,
    "media_next": None,
    "screenshot": None,
    "lock_screen": None,
    "tell_time": None,
    "cancel": None,
}

LABELS = (*INTENTS, OUT_OF_SCOPE)
