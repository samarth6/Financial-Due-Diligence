"""Tests for Layer 7, the shared room (src/dd/room.py) and its web server
(src/dd/server.py).

The room calls the real pipeline and the real Layer 6 actions through three
hooks in dd.room: run_answer, get_actions and warm_up. These tests swap those
hooks for fast stand-ins (pytest's monkeypatch, inside this file only), so the
room's rules - queue, live progress, voting - are checked in seconds without
loading models or calling Groq. The server code itself is unchanged.

Run from the project root:   pytest
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

import dd.room as room_mod
from dd.models import Answer, Citation, ConfidenceBreakdown
from dd.room import Room
from dd.server import create_app

# --- stand-ins for the pipeline and the actions ----------------------------------


def make_answer(question: str, answered: bool = True) -> Answer:
    if not answered:
        return Answer(query=question, answered=False, text="the filing does not say")
    return Answer(
        query=question,
        answered=True,
        text="Net income was $112,010 million [S1]",
        citations=[Citation(marker="[S1]", chunk_id="structured:x", source="AAPL | 10-K | FY2025 | income statement")],
        confidence=ConfidenceBreakdown(score=0.9, band="high", hybrid_agreement=0.0,
                                       rerank_strength=0.0, loop_cleanliness=1.0),
    )


class FakePipeline:
    """Reports two steps like the real one, then answers. `gate` lets a test
    hold a question 'running' to see what queues behind it."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.gate.set()
        self.asked: list[str] = []
        self.fail_on: set[str] = set()
        self.unanswered: set[str] = set()

    def __call__(self, question: str, on_progress) -> Answer:
        self.asked.append(question)
        on_progress("search", {"attempt": 1, "query": question})
        self.gate.wait(timeout=10)
        on_progress("write", {"passages": 1, "facts": 1})
        if question in self.fail_on:
            raise RuntimeError("Groq rate limit")
        return make_answer(question, answered=question not in self.unanswered)


class Draft(BaseModel):
    title: str


class FakeActions:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.problems: list[str] = []

    def table(self) -> dict:
        def propose(answer: Answer) -> Draft:
            return Draft(title=f"item for: {answer.query}")

        def validate(draft: Draft, answer: Answer) -> list[str]:
            return list(self.problems)

        def execute(draft: Draft, answer: Answer) -> str:
            self.executed.append(draft.title)
            return "written"

        return {"checklist": (propose, validate, execute), "flag": (propose, validate, execute)}


class FakeConn:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send_json(self, data: Any) -> None:
        self.messages.append(data)

    @property
    def last_state(self) -> dict[str, Any]:
        return self.messages[-1]["state"]


@pytest.fixture
def fakes(monkeypatch):
    pipeline, actions = FakePipeline(), FakeActions()
    monkeypatch.setattr(room_mod, "run_answer", pipeline)
    monkeypatch.setattr(room_mod, "get_actions", actions.table)
    monkeypatch.setattr(room_mod, "warm_up", lambda: None)
    return pipeline, actions


def run(coro):
    return asyncio.run(coro)


async def until(condition, timeout: float = 5.0) -> None:
    async def poll():
        while not condition():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(poll(), timeout)


async def started_room() -> Room:
    room = Room()
    await room.start()
    await until(lambda: room.status == "ready")
    return room


async def answered_question(room: Room, member: str, text: str = "What was net income in FY2025?") -> str:
    await room.ask(member, text)
    await until(lambda: any(h["text"] == text for h in room.history))
    return next(h["id"] for h in room.history if h["text"] == text)


# --- people ----------------------------------------------------------------------


def test_join_names_are_trimmed_and_kept_distinct(fakes):
    async def scenario():
        room = await started_room()
        a = await room.join(FakeConn(), "  Sam  ")
        b = await room.join(FakeConn(), "Sam")
        assert [room.members[a]["name"], room.members[b]["name"]] == ["Sam", "Sam (2)"]
        with pytest.raises(ValueError):
            await room.join(FakeConn(), "   ")
        await room.stop()
    run(scenario())


def test_everyone_receives_the_same_state_with_their_own_id(fakes):
    async def scenario():
        room = await started_room()
        c1, c2 = FakeConn(), FakeConn()
        a = await room.join(c1, "Ana")
        b = await room.join(c2, "Ben")
        assert c1.messages[-1]["you"] == a and c2.messages[-1]["you"] == b
        assert c1.last_state == c2.last_state
        assert [m["name"] for m in c1.last_state["members"]] == ["Ana", "Ben"]
        await room.stop()
    run(scenario())


# --- questions ---------------------------------------------------------------------


def test_question_length_is_checked(fakes):
    async def scenario():
        room = await started_room()
        m = await room.join(FakeConn(), "Ana")
        for bad in ["", "hi", "x" * (room_mod.MAX_QUESTION + 1)]:
            with pytest.raises(ValueError):
                await room.ask(m, bad)
        await room.stop()
    run(scenario())


def test_questions_queue_and_run_one_at_a_time_in_order(fakes):
    pipeline, _ = fakes

    async def scenario():
        room = await started_room()
        m = await room.join(FakeConn(), "Ana")
        pipeline.gate.clear()  # hold the first question "running"
        await room.ask(m, "first question?")
        await until(lambda: room.current is not None)
        await room.ask(m, "second question?")
        assert room.current["text"] == "first question?"
        assert [q["text"] for q in room.queue] == ["second question?"]

        pipeline.gate.set()
        await until(lambda: len(room.history) == 2)
        assert pipeline.asked == ["first question?", "second question?"]
        assert [h["text"] for h in room.history] == ["first question?", "second question?"]
        assert room.history[0]["answer"]["text"].startswith("Net income")
        await room.stop()
    run(scenario())


def test_live_steps_are_recorded_for_the_running_question(fakes):
    async def scenario():
        room = await started_room()
        m = await room.join(FakeConn(), "Ana")
        await answered_question(room, m)
        assert [s["stage"] for s in room.history[0]["steps"]] == ["search", "write"]
        await room.stop()
    run(scenario())


def test_a_late_step_is_never_filed_under_the_next_question(fakes):
    pipeline, _ = fakes

    async def scenario():
        room = await started_room()
        m = await room.join(FakeConn(), "Ana")
        first = await answered_question(room, m, "first question?")
        pipeline.gate.clear()
        await room.ask(m, "second question?")
        await until(lambda: room.current is not None)
        room._step_from_thread(first, "grade", {"attempt": 9})  # a straggler from question 1
        await asyncio.sleep(0.05)
        assert all(s["details"].get("attempt") != 9 for s in room.current["steps"])
        pipeline.gate.set()
        await room.stop()
    run(scenario())


def test_anyone_can_remove_a_waiting_question(fakes):
    pipeline, _ = fakes

    async def scenario():
        room = await started_room()
        m = await room.join(FakeConn(), "Ana")
        pipeline.gate.clear()
        await room.ask(m, "first question?")
        await until(lambda: room.current is not None)
        await room.ask(m, "second question?")
        await room.remove(m, room.queue[0]["id"])
        assert room.queue == []
        with pytest.raises(ValueError):
            await room.remove(m, "no-such-id")
        pipeline.gate.set()
        await room.stop()
    run(scenario())


def test_a_failed_question_is_reported_and_the_room_keeps_going(fakes):
    pipeline, _ = fakes
    pipeline.fail_on.add("broken question?")

    async def scenario():
        room = await started_room()
        m = await room.join(FakeConn(), "Ana")
        await room.ask(m, "broken question?")
        await room.ask(m, "working question?")
        await until(lambda: len(room.history) == 2)
        assert "rate limit" in room.history[0]["error"] and room.history[0]["answer"] is None
        assert room.history[1]["answer"] is not None
        await room.stop()
    run(scenario())


# --- actions: everyone present must approve ---------------------------------------


def test_action_runs_only_after_everyone_present_approves(fakes):
    _, actions = fakes

    async def scenario():
        room = await started_room()
        a = await room.join(FakeConn(), "Ana")
        b = await room.join(FakeConn(), "Ben")
        qid = await answered_question(room, a)
        await room.propose(a, "checklist", qid)
        action_id = room.pending_action["id"]
        assert room.pending_action["status"] == "voting"

        await room.vote(a, action_id, True)
        assert actions.executed == [], "ran before everyone approved"
        await room.vote(b, action_id, True)
        assert len(actions.executed) == 1
        assert room.pending_action is None and room.action_history[-1]["status"] == "done"
        await room.stop()
    run(scenario())


def test_one_rejection_stops_the_action(fakes):
    _, actions = fakes

    async def scenario():
        room = await started_room()
        a = await room.join(FakeConn(), "Ana")
        b = await room.join(FakeConn(), "Ben")
        qid = await answered_question(room, a)
        await room.propose(a, "checklist", qid)
        action_id = room.pending_action["id"]
        await room.vote(a, action_id, True)
        await room.vote(b, action_id, False)
        assert actions.executed == []
        assert room.action_history[-1]["status"] == "rejected by Ben"
        await room.stop()
    run(scenario())


def test_a_voter_who_leaves_is_no_longer_waited_on(fakes):
    _, actions = fakes

    async def scenario():
        room = await started_room()
        a = await room.join(FakeConn(), "Ana")
        b = await room.join(FakeConn(), "Ben")
        qid = await answered_question(room, a)
        await room.propose(a, "checklist", qid)
        await room.vote(a, room.pending_action["id"], True)
        await room.leave(b)
        assert len(actions.executed) == 1
        await room.stop()
    run(scenario())


def test_someone_who_joins_mid_vote_does_not_vote(fakes):
    async def scenario():
        room = await started_room()
        a = await room.join(FakeConn(), "Ana")
        qid = await answered_question(room, a)
        await room.join(FakeConn(), "Ben")
        b = await room.join(FakeConn(), "Cat")
        await room.leave(b)  # Cat comes and goes before the proposal
        await room.propose(a, "checklist", qid)
        late = await room.join(FakeConn(), "Dev")
        with pytest.raises(ValueError, match="joined after"):
            await room.vote(late, room.pending_action["id"], True)
        await room.stop()
    run(scenario())


def test_a_draft_that_breaks_the_rules_never_reaches_a_vote(fakes):
    _, actions = fakes
    actions.problems = ["answer confidence is LOW; too weak to act on"]

    async def scenario():
        room = await started_room()
        a = await room.join(FakeConn(), "Ana")
        qid = await answered_question(room, a)
        await room.propose(a, "checklist", qid)
        assert room.pending_action is None
        last = room.action_history[-1]
        assert last["status"] == "rejected by the rules" and last["voters"] == []
        assert actions.executed == []
        await room.stop()
    run(scenario())


def test_only_flag_can_act_on_an_unanswered_question(fakes):
    pipeline, _ = fakes
    pipeline.unanswered.add("unanswerable question?")

    async def scenario():
        room = await started_room()
        a = await room.join(FakeConn(), "Ana")
        qid = await answered_question(room, a, "unanswerable question?")
        with pytest.raises(ValueError, match="only 'flag'"):
            await room.propose(a, "checklist", qid)
        await room.propose(a, "flag", qid)
        assert room.pending_action["status"] == "voting"
        await room.stop()
    run(scenario())


def test_one_action_at_a_time(fakes):
    async def scenario():
        room = await started_room()
        a = await room.join(FakeConn(), "Ana")
        await room.join(FakeConn(), "Ben")
        qid = await answered_question(room, a)
        with pytest.raises(ValueError, match="no answer"):
            await room.propose(a, "flag", "no-such-question")
        with pytest.raises(ValueError, match="unknown action"):
            await room.propose(a, "launch-rockets", qid)
        await room.propose(a, "checklist", qid)
        with pytest.raises(ValueError, match="already being decided"):
            await room.propose(a, "flag", qid)
        await room.stop()
    run(scenario())


# --- the WebSocket server ------------------------------------------------------------


def _next(ws, msg_type: str, limit: int = 20) -> dict:
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] == msg_type:
            return msg
    raise AssertionError(f"no '{msg_type}' message")


def test_server_health_and_join(fakes):
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["members"] == 0
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "join", "name": "Ana"})
            state = _next(ws, "state")["state"]
            assert [m["name"] for m in state["members"]] == ["Ana"]


def test_server_turns_bad_messages_into_errors_without_disconnecting(fakes):
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ask", "text": "too early?"})
        assert "join with a name first" in _next(ws, "error")["message"]

        ws.send_text("not json at all")
        _next(ws, "error")
        ws.send_text(json.dumps(["a", "list"]))
        assert "JSON object" in _next(ws, "error")["message"]

        ws.send_json({"type": "join", "name": "Ana"})
        _next(ws, "state")
        ws.send_json({"type": "vote", "action_id": "x", "approve": "false"})
        assert "must be true or false" in _next(ws, "error")["message"]

        ws.send_json({"type": "ask", "text": "still connected and working?"})
        state = _next(ws, "state")["state"]
        assert state["members"][0]["name"] == "Ana"


# --- nothing else changed --------------------------------------------------------------


def test_pipeline_progress_is_optional_and_never_breaks_the_answer():
    """Layers 4-5 gained an optional on_progress argument. Existing callers
    (dd answer, dd heal, the eval) pass nothing; a broken listener is ignored."""
    import inspect

    from dd.generation import answer, generate_answer
    from dd.self_healing import emit, self_heal

    for fn in (answer, generate_answer, self_heal):
        param = inspect.signature(fn).parameters["on_progress"]
        assert param.default is None

    def broken_listener(stage: str, details: dict) -> None:
        raise RuntimeError("listener crashed")

    emit(None, "search", attempt=1)
    emit(broken_listener, "search", attempt=1)  # must not raise
