"""The dashboard's backend: a JSON API over the calendar, and the built React app on the same port.

    python -m aiforme dashboard          # http://127.0.0.1:8765

Local only: it listens on 127.0.0.1 and has no accounts. Every change to the calendar is also published on the bus
as `calendar.changed`, so modules that cache the calendar can refresh.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aiforme.calendar import Calendar
from aiforme.core import Event, EventBus

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
HOST, PORT = "127.0.0.1", 8765
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]     # the Vite dev server

Kind = Literal["meeting", "shift", "other"]


class EventIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    start: datetime
    end: datetime
    kind: Kind = "other"
    notes: str = Field(default="", max_length=2000)


class EventPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    start: datetime | None = None
    end: datetime | None = None
    kind: Kind | None = None
    notes: str | None = Field(default=None, max_length=2000)


def create_app(calendar: Calendar, bus_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="AI For Me", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])

    def changed(action: str, event_id: int) -> None:
        if bus_path is None:
            return
        with EventBus(bus_path) as bus:         # a connection per request: the server runs handlers in threads
            bus.publish(Event("calendar.changed", {"action": action, "id": event_id}, source="dashboard"))

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/mode")
    def mode() -> dict:
        """The mode the autopilot last switched to, and why."""
        if bus_path is None:
            return {"mode": None}
        with EventBus(bus_path) as bus:
            last = bus.recent(1, "mode.changed")
        if not last:
            return {"mode": None}
        p = last[0].payload
        return {"mode": p.get("mode"), "reason": p.get("reason"), "since": last[0].created_at}

    @app.get("/api/events")
    def list_events(start: datetime, end: datetime) -> list[dict]:
        start, end = _naive(start), _naive(end)
        if end <= start:
            raise HTTPException(422, f"the range must end after it starts: {start} – {end}")
        return [e.to_json() for e in calendar.between(start, end)]

    @app.post("/api/events", status_code=201)
    def add_event(body: EventIn) -> dict:
        try:
            ev = calendar.add(body.title, _naive(body.start), _naive(body.end), body.kind, body.notes)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        changed("added", ev.id)
        return ev.to_json()

    @app.patch("/api/events/{event_id}")
    def update_event(event_id: int, body: EventPatch) -> dict:
        fields = {k: _naive(v) if isinstance(v, datetime) else v
                  for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
        try:
            ev = calendar.update(event_id, **fields)
        except KeyError as exc:
            raise HTTPException(404, f"no event {event_id}") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        changed("updated", ev.id)
        return ev.to_json()

    @app.delete("/api/events/{event_id}", status_code=204)
    def delete_event(event_id: int) -> Response:
        if not calendar.delete(event_id):
            raise HTTPException(404, f"no event {event_id}")
        changed("deleted", event_id)
        return Response(status_code=204)

    if WEB_DIST.exists():                       # the built frontend; `npm run dev` serves it instead in development
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            file = (WEB_DIST / path).resolve()
            if path and file.is_file() and WEB_DIST in file.parents:
                return FileResponse(file)
            return FileResponse(WEB_DIST / "index.html")

    return app


def _naive(t: datetime) -> datetime:
    """The calendar keeps local times; a browser may send UTC with an offset."""
    return t.astimezone().replace(tzinfo=None) if t.tzinfo else t


def serve(calendar: Calendar, bus_path: Path | None = None, host: str = HOST, port: int = PORT):
    """A uvicorn server for `create_app`, not yet started: `.run()` blocks, `.should_exit = True` stops it."""
    import uvicorn
    config = uvicorn.Config(create_app(calendar, bus_path), host=host, port=port, log_level="warning")
    return uvicorn.Server(config)
