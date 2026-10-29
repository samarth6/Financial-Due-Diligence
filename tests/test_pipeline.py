"""End-to-end tests for Layers 4 and 5: the REAL self_heal() and answer().

Only the outside world is replaced with stand-ins - search (hybrid_retrieve),
the database lookup (get_structured_facts) and Groq (get_groq_client) - so
these run in a second, offline, on any machine. Everything in between is the
real code: grading, fact filtering, retries, best-attempt tracking, early
give-up, error handling, citation checks, and Layer 7's progress reports.

Why this file exists: a merge once replaced parts of self_healing.py and
generation.py with older code. Every question crashed, yet all other tests
passed - they never ran the real pipeline. Each test below would have failed.

Run from the project root:   pytest
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

import dd.generation as gen
import dd.self_healing as sh
from dd.config import get_config
from dd.models import Chunk, RetrievedChunk

CFG = get_config().self_healing


# --- stand-ins for the outside world ---------------------------------------------


def passages(tag: str, *grades: float) -> list[RetrievedChunk]:
    """Retrieved passages whose text carries the grade the fake grader gives."""
    return [
        RetrievedChunk(chunk=Chunk(
            chunk_id=f"{tag}-{i}", text=f"passage {tag}-{i} grade={g}",
            company="Apple Inc.", ticker="AAPL", form="10-K", fiscal_year="FY2025",
            item="Item 1A", section="Risk Factors",
        ))
        for i, g in enumerate(grades)
    ]


def fact(label: str, value: float, table: str = "income_statement") -> dict:
    return {"table": table, "label": label, "year": "2025", "value": value,
            "concept": "us-gaap_X", "_matched_via": "keyword"}


class FakeGroq:
    """Answers the three kinds of Groq call the pipeline makes, by name:
    grading (chunk_grades), answering (grounded_answer), and rewording (no
    schema). `fail` makes the named call raise, like a rate limit."""

    def __init__(self, fact_grades: dict[str, float] | None = None,
                 answer: dict | None = None, fail: str | None = None,
                 unreadable_grades: bool = False, shuffle_grades: bool = False) -> None:
        self.fact_grades = fact_grades or {}
        self.answer = answer or {"answerable": True, "answer": "Tariffs could hurt margins [1].", "citations": ["[1]"]}
        self.fail = fail
        self.unreadable_grades = unreadable_grades
        self.shuffle_grades = shuffle_grades  # list grades in reverse order, as a real model sometimes does
        self.rewrites = 0
        self.prompts: dict[str, list[str]] = {}  # call kind -> prompts sent, in order
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        fmt = kwargs.get("response_format")
        kind = fmt["json_schema"]["name"] if fmt else "rewrite"
        self.prompts.setdefault(kind, []).append(kwargs["messages"][0]["content"])
        if kind == self.fail:
            raise RuntimeError("rate limit exceeded")
        if kind == "chunk_grades":
            grades = [{"item": marker, "relevance": self._grade(item), "reason": "test"}
                      for marker, item in self._items(kwargs["messages"][0]["content"])]
            if self.shuffle_grades:
                grades.reverse()
            content = "not json at all" if self.unreadable_grades else json.dumps({"grades": grades})
        elif kind == "grounded_answer":
            content = json.dumps(self.answer)
        else:
            self.rewrites += 1
            content = f"reworded question {self.rewrites}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    @staticmethod
    def _items(prompt: str) -> list[tuple[str, str]]:
        """(marker, text) for each item, e.g. ("[0]", "(Item 1A / ...) passage a1-0 grade=0.9")."""
        return re.findall(r"^(\[S?\d+\]) (.+)$", prompt, flags=re.MULTILINE)

    def _grade(self, item: str) -> float:
        if m := re.search(r"grade=([\d.]+)", item):
            return float(m.group(1))
        return next((g for label, g in self.fact_grades.items() if item.startswith(label)), 0.0)


@pytest.fixture
def world(monkeypatch):
    """Wire the stand-ins in. Returns a setter for what each attempt retrieves,
    what the database finds, and how Groq behaves."""
    state = {"retrievals": [], "facts": [], "groq": FakeGroq(), "calls": 0}

    def fake_retrieve(query, cfg):
        attempt = state["calls"]
        state["calls"] += 1
        return state["retrievals"][min(attempt, len(state["retrievals"]) - 1)]

    monkeypatch.setattr(sh, "hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr(sh, "get_structured_facts", lambda q, cfg: state["facts"])
    monkeypatch.setattr(sh, "get_groq_client", lambda cfg: state["groq"])
    monkeypatch.setattr(gen, "get_groq_client", lambda cfg: state["groq"])

    def setup(retrievals, facts=None, groq=None):
        state["retrievals"], state["facts"] = retrievals, facts or []
        state["groq"] = groq or FakeGroq()
        state["calls"] = 0
    return setup


def stages(events: list) -> list[str]:
    return [stage for stage, _ in events]


def test_config_assumptions():
    """The scenarios below rely on these settings (from config.yaml)."""
    assert CFG.relevance_threshold == 0.5 and CFG.strong_chunk_threshold == 0.8
    assert CFG.min_relevant_chunks == 2 and CFG.max_attempts == 3 and CFG.give_up_below == 0.2


# --- Layer 4 -----------------------------------------------------------------


def test_good_evidence_succeeds_on_the_first_attempt(world):
    world([passages("a1", 0.9, 0.6, 0.1)])
    events = []
    outcome = sh.self_heal("tariff risks?", on_progress=lambda s, d: events.append((s, d)))
    assert outcome.succeeded and len(outcome.attempts) == 1
    assert [g.relevance for g in outcome.relevant_chunks] == [0.9, 0.6]
    assert stages(events) == ["lookup", "search", "grade", "graded"]


def test_database_facts_are_graded_and_only_relevant_ones_kept(world):
    world([passages("a1", 0.1, 0.1)],
          facts=[fact("Net income", 112_010e6), fact("Total assets", 359_241e6, "balance_sheet")],
          groq=FakeGroq(fact_grades={"Net income": 0.9, "Total assets": 0.0}))
    events = []
    outcome = sh.self_heal("net income FY2025?", on_progress=lambda s, d: events.append((s, d)))
    assert outcome.succeeded, "one strong, relevant fact is enough evidence"
    assert [f["label"] for f in outcome.structured_facts] == ["Net income"], "irrelevant fact must be dropped"
    graded = dict(events)["graded"]
    assert (graded["facts_relevant"], graded["facts_total"]) == (1, 2)


def test_a_refusal_keeps_the_best_attempt_not_the_last(world):
    world([passages("a1", 0.4, 0.3), passages("a2", 0.6, 0.4), passages("a3", 0.3, 0.3)])
    events = []
    outcome = sh.self_heal("hard question?", on_progress=lambda s, d: events.append((s, d)))
    assert not outcome.succeeded and len(outcome.attempts) == 3
    assert {g.retrieved.chunk.chunk_id for g in outcome.chunks} == {"a2-0", "a2-1"}
    assert outcome.final_query == "reworded question 1"
    assert stages(events).count("rewrite") == 2


def test_gives_up_early_when_nothing_comes_close(world):
    world([passages("a1", 0.05, 0.1)])
    events = []
    outcome = sh.self_heal("FY2027 projections?", on_progress=lambda s, d: events.append((s, d)))
    assert not outcome.succeeded and len(outcome.attempts) == 1
    assert stages(events)[-1] == "give_up"


def test_stops_when_rewording_finds_the_same_passages(world):
    same = passages("x", 0.3, 0.3)
    world([same, same])
    events = []
    outcome = sh.self_heal("question?", on_progress=lambda s, d: events.append((s, d)))
    assert not outcome.succeeded and len(outcome.attempts) == 2
    assert stages(events)[-1] == "stagnated"


def test_the_grader_gets_the_scale_headings_and_whole_passage_rule(world):
    """The scale's steps must be the thresholds self_heal() decides with, and
    each passage must arrive with where it sits in the filing."""
    groq = FakeGroq()
    retrieved = passages("a1", 0.9)
    retrieved[0].chunk.subsection = "CONSOLIDATED BALANCE SHEETS"
    world([retrieved + passages("a2", 0.6)], groq=groq)
    sh.self_heal("total assets?")
    prompt = groq.prompts["chunk_grades"][0]

    assert "[0] (Item 1A / CONSOLIDATED BALANCE SHEETS) passage a1-0" in prompt
    assert "[1] (Item 1A / Risk Factors) passage a2-0" in prompt, "no subsection: falls back to the section"
    for score in (CFG.strong_chunk_threshold, CFG.relevance_threshold, CFG.give_up_below):
        assert f"\n{score:g} - " in prompt
    assert "WHOLE passage" in prompt


def test_grades_are_matched_by_marker_not_position(world):
    """A real model once listed correct grades in the wrong order; matched by
    position, the passage holding the answer got the score of the one without."""
    world([passages("a1", 0.9, 0.1, 0.6)],
          facts=[fact("Net income", 112_010e6)],
          groq=FakeGroq(fact_grades={"Net income": 0.7}, shuffle_grades=True))
    outcome = sh.self_heal("question?")
    assert [g.relevance for g in outcome.chunks] == [0.9, 0.1, 0.6]
    assert outcome.structured_facts[0]["_relevance"] == 0.7


def test_an_item_the_grader_skipped_counts_as_ungraded(world, monkeypatch):
    groq = FakeGroq()
    real = groq._create

    def drop_item_1(**kwargs):
        resp = real(**kwargs)
        if kwargs.get("response_format"):
            body = json.loads(resp.choices[0].message.content)
            body["grades"] = [g for g in body["grades"] if g["item"] != "[1]"]
            resp.choices[0].message.content = json.dumps(body)
        return resp

    groq.chat.completions.create = drop_item_1
    world([passages("a1", 0.9, 0.9)], groq=groq)
    outcome = sh.self_heal("question?")
    assert [(g.relevance, g.reason) for g in outcome.chunks] == [(0.9, "test"), (0.0, "ungraded")]


@pytest.mark.parametrize("written, expected", [
    ("[0]", "0"), ("0", "0"), ("Item [2]", "2"), ("[S1]", "S1"), ("s1", "S1"), ("", ""), (None, ""),
])
def test_markers_are_read_however_the_model_writes_them(written, expected):
    assert sh._marker(written) == expected


def test_an_unreadable_grader_reply_refuses_instead_of_crashing(world):
    world([passages("a1", 0.9, 0.9)], groq=FakeGroq(unreadable_grades=True))
    outcome = sh.self_heal("question?")
    assert not outcome.succeeded
    assert all(g.relevance == 0.0 for g in outcome.chunks)


# --- Layers 4 + 5 together (answer) ----------------------------------------------


def test_full_answer_with_progress_reports(world):
    world([passages("a1", 0.9, 0.6)])
    events = []
    a = gen.answer("tariff risks?", on_progress=lambda s, d: events.append((s, d)))
    assert a.answered and [c.marker for c in a.citations] == ["[1]"]
    assert stages(events) == ["lookup", "search", "grade", "graded", "write"]


def test_answer_works_without_progress_reports(world):
    """Existing callers (dd answer, the eval) pass nothing."""
    world([passages("a1", 0.9, 0.6)])
    assert gen.answer("tariff risks?").answered


def test_a_broken_progress_listener_never_breaks_the_answer(world):
    world([passages("a1", 0.9, 0.6)])

    def broken(stage, details):
        raise RuntimeError("listener crashed")

    assert gen.answer("tariff risks?", on_progress=broken).answered


def test_groq_failing_in_layer_4_gives_a_plain_message(world):
    world([passages("a1", 0.9)], groq=FakeGroq(fail="chunk_grades"))
    a = gen.answer("question?")
    assert not a.answered and a.text.startswith("Model call failed")


def test_groq_failing_in_layer_5_gives_a_plain_message(world):
    world([passages("a1", 0.9, 0.6)], groq=FakeGroq(fail="grounded_answer"))
    a = gen.answer("question?")
    assert not a.answered and a.text.startswith("Model call failed")
    assert a.outcome is not None, "the retrieved evidence is kept for inspection"


def test_a_refusal_from_layer_4_is_passed_through(world):
    world([passages("a1", 0.05)])
    a = gen.answer("FY2027 projections?")
    assert not a.answered and "relevant chunk" in a.text
