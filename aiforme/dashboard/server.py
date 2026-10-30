"""The dashboard's backend: a JSON API over the calendar and the heard commands, and the built React app.

    python -m aiforme dashboard          # http://127.0.0.1:8765

Local only: it listens on 127.0.0.1 and has no accounts. Every change to the calendar is also published on the bus
as `calendar.changed`, so modules that cache the calendar can refresh. Commands are read from the bus; labels on
them and the retraining they feed live in aiforme/active_learning.
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aiforme.active_learning.retrain import NothingToLearn, current_threshold, retrain_and_announce
from aiforme.active_learning.store import Feedback, LabelLocked
from aiforme.calendar import Calendar
from aiforme.core import Event, EventBus
from aiforme.voice_orchestrator.intents import INTENTS, LABELS, OUT_OF_SCOPE
from aiforme.voice_orchestrator.nlu import normalize

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
HOST, PORT = "127.0.0.1", 8765
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]     # the Vite dev server

Kind = Literal["meeting", "shift", "other"]
COMMAND_TOPICS = ("intent.recognized", "intent.unknown")
AUTO_RETRAIN_EVERY = 20         # new labels since the last retraining that start the next one on their own


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


class LabelIn(BaseModel):
    intent: str
    slots: dict[str, str] = Field(default_factory=dict)


def create_app(calendar: Calendar, bus_path: Path | None = None, feedback_path: Path | None = None,
               embedder: str = "e5", auto_retrain_every: int = AUTO_RETRAIN_EVERY) -> FastAPI:
    """`auto_retrain_every` 0 leaves retraining to the button."""
    app = FastAPI(title="AI For Me", docs_url="/api/docs", openapi_url="/api/openapi.json")
    # one retraining at a time; the thread is kept so tests can wait for it
    app.state.retraining = {"thread": None, "error": None, "automatic": False, "lock": threading.Lock()}
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

    def stores() -> tuple[EventBus, Feedback]:
        if feedback_path is None or bus_path is None:
            raise HTTPException(503, "the dashboard runs without the bus or the feedback store")
        return EventBus(bus_path), Feedback(feedback_path)

    @app.get("/api/commands")
    def commands(limit: int = 50, order: Literal["recent", "uncertain"] = "recent") -> list[dict]:
        """The commands the assistant heard, with my label on each if there is one. `recent`: newest first.
        `uncertain`: the unlabelled ones first, closest to the out-of-scope threshold first — the calls the model
        nearly made the other way, where a label teaches it the most — then the labelled ones, newest first."""
        limit = min(max(limit, 1), 500)
        window = limit if order == "recent" else 500
        bus, fb = stores()
        with bus, fb:
            # intent.* also holds intent.model_updated: read a little more than asked for
            events = [e for e in reversed(bus.recent(window * 2, "intent.*")) if e.topic in COMMAND_TOPICS][:window]
            labels = fb.get([e.id for e in events])
            fallback = current_threshold(fb)
        rows = [{**_command(e), "label": labels[e.id].to_json() if e.id in labels else None} for e in events]
        for row in rows:
            threshold = row.pop("threshold") or fallback
            row["margin"] = None if threshold is None else round(abs(row["confidence"] - threshold), 3)
        if order == "uncertain":
            rows.sort(key=_doubt)
        return rows[:limit]

    @app.put("/api/commands/{event_id}/label")
    def label(event_id: int, body: LabelIn) -> dict:
        bus, fb = stores()
        with bus, fb:
            event = bus.get(event_id)
            if event is None or event.topic not in COMMAND_TOPICS:
                raise HTTPException(404, f"no command {event_id}")
            cmd = _command(event)
            slots = _check_slots(body.intent, body.slots, cmd["text"])
            try:
                lab = fb.label(event_id, cmd["text"], cmd["intent"], cmd["slots"], cmd["confidence"], cmd["model"],
                               event.created_at, body.intent, slots)
            except LabelLocked as exc:
                raise HTTPException(409, str(exc)) from exc
            due = auto_retrain_every and fb.pending_since_last_version() >= auto_retrain_every
        if due:
            start(automatic=True)
        return lab.to_json()

    @app.delete("/api/commands/{event_id}/label", status_code=204)
    def unlabel(event_id: int) -> Response:
        bus, fb = stores()
        with bus, fb:
            try:
                removed = fb.remove(event_id)
            except LabelLocked as exc:
                raise HTTPException(409, str(exc)) from exc
        if not removed:
            raise HTTPException(404, f"command {event_id} has no label")
        return Response(status_code=204)

    @app.get("/api/learning")
    def learning() -> dict:
        """What a retraining has to work with, every version so far, and how each did on the commands it served."""
        bus, fb = stores()
        with bus, fb:
            pending = fb.labels(pending=True)
            versions, live, current = fb.versions(), fb.live_accuracy(), fb.current_version()
            new_labels = fb.pending_since_last_version()
        thread = app.state.retraining["thread"]
        return {
            "version": current, "embedder": embedder, "intents": {**INTENTS, OUT_OF_SCOPE: None},
            "pending": {"confirmed": sum(lab.verdict == "confirmed" for lab in pending),
                        "corrected": sum(lab.verdict == "corrected" for lab in pending)},
            "retraining": thread is not None and thread.is_alive(), "error": app.state.retraining["error"],
            "automatic": app.state.retraining["automatic"],
            "auto": {"every": auto_retrain_every, "new": new_labels},
            "versions": [v.to_json() for v in versions], "live": live,
        }

    @app.post("/api/learning/retrain", status_code=202)
    def start_retrain() -> dict:
        if running():
            raise HTTPException(409, "a retraining is already running")
        bus, fb = stores()
        with bus, fb:
            if not fb.labels(pending=True):
                raise HTTPException(409, "nothing to learn: no new labels")
        start(automatic=False)
        return {"retraining": True}

    def running() -> bool:
        thread = app.state.retraining["thread"]
        return thread is not None and thread.is_alive()

    def start(automatic: bool) -> None:
        """A retraining in a thread of its own; a second one while it runs is not started."""
        state = app.state.retraining
        with state["lock"]:
            if running():
                return
            state.update(error=None, automatic=automatic,
                         thread=threading.Thread(target=work, name="retrain", daemon=True))
            state["thread"].start()

    def work():
        try:
            with EventBus(bus_path) as bus, Feedback(feedback_path) as fb:
                retrain_and_announce(fb, bus, embedder)
        except NothingToLearn:
            pass
        except Exception as exc:                # shown on the page; the next attempt starts clean
            app.state.retraining["error"] = repr(exc)

    if WEB_DIST.exists():                       # the built frontend; `npm run dev` serves it instead in development
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            file = (WEB_DIST / path).resolve()
            if path and file.is_file() and WEB_DIST in file.parents:
                return FileResponse(file)
            return FileResponse(WEB_DIST / "index.html")

    return app


def _command(e: Event) -> dict:
    p = e.payload
    return {"id": e.id, "heard_at": e.created_at, "text": str(p.get("text", "")),
            "intent": OUT_OF_SCOPE if e.topic == "intent.unknown" else str(p.get("intent")),
            "confidence": float(p.get("confidence", 0.0)), "slots": p.get("slots") or {},
            "best_guess": p.get("best_guess"), "model": int(p.get("model", 0)), "threshold": p.get("threshold")}


def _doubt(row: dict) -> tuple:
    """Sort key: unlabelled before labelled; among the unlabelled the narrowest margin first (without a known
    threshold, the lowest confidence); then the newest."""
    if row["label"] is not None:
        return (1, 0.0, -row["heard_at"])
    closeness = row["margin"] if row["margin"] is not None else row["confidence"]
    return (0, closeness, -row["heard_at"])


def _check_slots(intent: str, slots: dict[str, str], text: str) -> dict[str, str]:
    """The slot the intent takes and nothing else. Its value must be words of the command: the slot extractor
    learns the words around it."""
    if intent not in LABELS:
        raise HTTPException(422, f"unknown intent {intent!r}")
    name = INTENTS.get(intent)
    if not name:
        return {}
    value = " ".join(slots.get(name, "").split())
    if not value:
        raise HTTPException(422, f"{intent} needs its {name}")
    if not normalize(value) or f" {normalize(value)} " not in f" {normalize(text)} ":
        raise HTTPException(422, f"the {name} must be words of the command: {value!r} is not in {text!r}")
    return {name: value}


def _naive(t: datetime) -> datetime:
    """The calendar keeps local times; a browser may send UTC with an offset."""
    return t.astimezone().replace(tzinfo=None) if t.tzinfo else t


def serve(calendar: Calendar, bus_path: Path | None = None, feedback_path: Path | None = None, embedder: str = "e5",
          host: str = HOST, port: int = PORT):
    """A uvicorn server for `create_app`, not yet started: `.run()` blocks, `.should_exit = True` stops it."""
    import uvicorn
    config = uvicorn.Config(create_app(calendar, bus_path, feedback_path, embedder), host=host, port=port,
                            log_level="warning")
    return uvicorn.Server(config)
