"""Layer 7: the web server - networking only.

Serves the team's web page and one WebSocket, /ws, that connects each
browser to the shared room. All the rules (queue, live progress, the vote on
actions) live in room.py; this file only turns WebSocket messages into Room
calls and Room errors into messages.

Messages a browser sends (JSON objects):
    {"type": "join",    "name": "Sam"}                       - must come first
    {"type": "ask",     "text": "What risks ...?"}
    {"type": "remove",  "question_id": "..."}               - from the queue
    {"type": "propose", "kind": "checklist|flag|memo", "question_id": "..."}
    {"type": "vote",    "action_id": "...", "approve": true|false}

Messages the server sends:
    {"type": "state", "you": "<your member id>", "state": {...}}  - after every change
    {"type": "error", "message": "..."}                           - only to the sender

Start it with `dd serve`.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT
from .room import Room

WEB_DIST = PROJECT_ROOT / "web" / "dist"


def _field(msg: dict[str, Any], key: str) -> str:
    value = msg.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be text")
    return value


async def _handle(room: Room, member_id: str, msg: dict[str, Any]) -> None:
    kind = msg.get("type")
    if kind == "ask":
        await room.ask(member_id, _field(msg, "text"))
    elif kind == "remove":
        await room.remove(member_id, _field(msg, "question_id"))
    elif kind == "propose":
        await room.propose(member_id, _field(msg, "kind"), _field(msg, "question_id"))
    elif kind == "vote":
        approve = msg.get("approve")
        if not isinstance(approve, bool):
            # Only a real true/false counts: the text "false" must never
            # be read as a yes.
            raise ValueError("'approve' must be true or false")
        await room.vote(member_id, _field(msg, "action_id"), approve)
    elif kind == "join":
        raise ValueError("you have already joined")
    else:
        raise ValueError(f"unknown message type '{kind}'")


def create_app() -> FastAPI:
    room = Room()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await room.start()
        yield
        await room.stop()

    app = FastAPI(title="Financial due-diligence room", lifespan=lifespan)
    app.state.room = room

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": room.status, "members": len(room.members)}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        member_id: str | None = None
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("a message must be a JSON object")
                    if member_id is None:
                        if msg.get("type") != "join":
                            raise ValueError("join with a name first")
                        member_id = await room.join(ws, _field(msg, "name"))
                    else:
                        await _handle(room, member_id, msg)
                except (ValueError, json.JSONDecodeError) as exc:
                    # A bad message is answered, never fatal: the sender
                    # stays connected and nobody else is affected.
                    await ws.send_json({"type": "error", "message": str(exc)})
        except WebSocketDisconnect:
            pass
        finally:
            if member_id is not None:
                await room.leave(member_id)

    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:
        @app.get("/")
        async def no_frontend() -> PlainTextResponse:
            return PlainTextResponse(
                "The web app hasn't been built yet. From the project root run:\n"
                "  cd web\n  npm install\n  npm run build\n"
                "then restart `dd serve`."
            )

    return app
