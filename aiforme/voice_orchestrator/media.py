"""What is playing, and turning the sound down while listening — the Windows APIs behind both.

Media: the system media transport controls, the same session Windows shows above the volume flyout. They say
whether a video or a song is playing and in which app, and pause or resume that exact session, so «пауза» never
toggles a paused video back on the way the play/pause key does.

Ducking: after the wake word the master volume drops to `level` (as Alexa and Siri do) and comes back once the
command is recorded. A video at full volume next to the microphone otherwise drowns the command.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

PLAYING, PAUSED = 4, 5          # GlobalSystemMediaTransportControlsSessionPlaybackStatus


@dataclass(frozen=True)
class MediaState:
    app: str
    title: str
    playing: bool


class MediaControl:
    def state(self) -> MediaState | None:
        return asyncio.run(self._state())

    def pause(self) -> MediaState | None:
        """Pauses what is playing; returns what it was, or None if nothing was playing."""
        return asyncio.run(self._act("pause"))

    def play(self) -> MediaState | None:
        """Resumes the current session; returns it, or None if there is none or it is already playing."""
        return asyncio.run(self._act("play"))

    def next(self) -> MediaState | None:
        return asyncio.run(self._act("next"))

    async def _session(self):
        from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as Manager
        manager = await Manager.request_async()
        return manager.get_current_session()

    async def _state(self, session=None) -> MediaState | None:
        session = session or await self._session()
        if session is None:
            return None
        props = await session.try_get_media_properties_async()
        status = session.get_playback_info().playback_status
        app = session.source_app_user_model_id.removesuffix(".exe")
        return MediaState(app, props.title or "", status == PLAYING)

    async def _act(self, action: str) -> MediaState | None:
        session = await self._session()
        if session is None:
            return None
        state = await self._state(session)
        if action == "pause":
            if not state.playing:
                return None
            await session.try_pause_async()
        elif action == "play":
            if state.playing:
                return None
            await session.try_play_async()
        else:
            await session.try_skip_next_async()
        return state


class VolumeDucker:
    """Lowers the master volume to `level` and puts it back; safe to call restore() any number of times."""

    def __init__(self, level: float = 0.10):
        from pycaw.pycaw import AudioUtilities
        self.level = level
        self._volume = AudioUtilities.GetSpeakers().EndpointVolume
        self._saved: float | None = None

    def duck(self) -> None:
        if self._saved is not None:
            return
        try:
            current = self._volume.GetMasterVolumeLevelScalar()
            if current > self.level:
                self._volume.SetMasterVolumeLevelScalar(self.level, None)
                self._saved = current
        except Exception:                       # a headset unplugged mid-session must not stop the listener
            log.exception("could not turn the volume down")

    def restore(self) -> None:
        if self._saved is None:
            return
        try:
            self._volume.SetMasterVolumeLevelScalar(self._saved, None)
        except Exception:
            log.exception("could not restore the volume")
        finally:
            self._saved = None
