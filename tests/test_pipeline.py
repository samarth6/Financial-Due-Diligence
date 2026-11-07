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
from dd.models import Chunk, GradedChunk, RetrievedChunk
from dd.models import Chunk, RetrievedChunk
from dd.structured import describe_fact, match_structured_query, segments_mentioned

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
                 answer: dict | list[dict] | None = None, fail: str | None = None,
                 unreadable_grades: bool = False, shuffle_grades: bool = False,
                 parts: list[str] | None = None) -> None:
        self.fact_grades = fact_grades or {}
        # A list is replied in turn, one per answer call.
        self.answer = answer or {"answerable": True, "answer": "Tariffs could hurt margins [1].", "citations": ["[1]"]}
        self.parts = parts or []  # what splitting the question returns
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
            content = json.dumps(self.answer.pop(0) if isinstance(self.answer, list) else self.answer)
        elif kind == "question_parts":
            content = json.dumps({"parts": self.parts})
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
    # Figures have their own scale: sharing the passages' one, each figure of a
    # comparison was graded 0.5, "one of several things asked for".
    assert "Figures, on this scale instead:" in prompt
    assert "company-wide total is not a figure for one product or segment" in prompt


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


# --- citation markers in the answer text ---------------------------------------
# Eval run on 5c78bb0: EPS, operating cash and iPhone revenue had the right
# number, [S1] in the citations list, but no marker in the text - and HIGH.


def test_one_sentence_answer_without_markers_gets_its_citations(world):
    world([passages("a1", 0.1)], facts=[fact("iPhone", 209_586e6)],
          groq=FakeGroq(fact_grades={"iPhone": 0.9},
                        answer={"answerable": True, "answer": "$209,586,000,000.", "citations": ["[S1]"]}))
    a = gen.answer("iPhone revenue FY2025?")
    assert a.text == "$209,586,000,000 [S1].", "a single sentence can only come from its cited sources"
    assert a.confidence.band == "high"


def test_longer_answer_without_markers_is_kept_below_high(world):
    world([passages("a1", 0.1)], facts=[fact("Net income", 112_010e6)],
          groq=FakeGroq(fact_grades={"Net income": 0.9},
                        answer={"answerable": True,
                                "answer": "Net income was $112,010,000,000. That was a record year.",
                                "citations": ["[S1]"]}))
    a = gen.answer("net income FY2025?")
    assert "[S1]" not in a.text, "with two sentences we can't tell which one the source backs"
    assert a.confidence.band != "high"
    assert any("not tied to" in n for n in a.confidence.notes)


def test_answer_with_markers_is_left_alone(world):
    world([passages("a1", 0.1)], facts=[fact("Net income", 112_010e6)],
          groq=FakeGroq(fact_grades={"Net income": 0.9},
                        answer={"answerable": True, "answer": "Net income was $112,010,000,000 [S1].",
                                "citations": ["[S1]"]}))
    a = gen.answer("net income FY2025?")
    assert a.text == "Net income was $112,010,000,000 [S1]." and a.confidence.band == "high"



# --- the number check ------------------------------------------------------------
# Eval run: two correct answers were capped LOW (0.44) as "number not found".


def _no_number_warning(a) -> bool:
    return not any("not found in cited sources" in n for n in a.confidence.notes)


def test_figures_restated_in_millions_are_recognised(world):
    facts = [fact("Total assets", 359_241e6, "balance_sheet"),
             fact("Total liabilities", 285_508e6, "balance_sheet"),
             fact("Total shareholders equity", 73_733e6, "balance_sheet")]
    world([passages("a1", 0.1)], facts=facts,
          groq=FakeGroq(fact_grades={"Total": 0.9}, answer={
              "answerable": True, "citations": ["[S1]", "[S2]", "[S3]"],
              "answer": "Yes, total assets of $359,241,000,000 [S1] equal total liabilities of "
                        "$285,508,000,000 [S2] plus total shareholders' equity of $73,733,000,000 [S3] "
                        "(285,508 + 73,733 = 359,241)."}))
    a = gen.answer("Does the balance sheet balance?")
    assert _no_number_warning(a) and a.confidence.band == "high"


WEARABLES_TABLE = ("(in millions) Wearables, Home and Accessories $ 35,686 (4)% $ 37,005 "
                   "Diluted earnings per share $ 7.46 $ 6.08 grade=0.9")


def _answer_from_table(world, answer: str):
    p = passages("a1", 0.9, 0.9)
    # "grade=0.9" is what the fake grader reads, so the passage stays relevant.
    p[0].chunk.text = WEARABLES_TABLE
    world([p], groq=FakeGroq(answer={"answerable": True, "citations": ["[1]"], "answer": answer}))
    return gen.answer("Why did Wearables net sales decrease?")


def test_differences_of_passage_numbers_are_allowed(world):
    a = _answer_from_table(world, "Net sales fell from $37,005 million in 2024 to $35,686 million in 2025, "
                                  "a decrease of $1,319 million [1].")
    assert _no_number_warning(a)


@pytest.mark.parametrize("answer", [
    "Net sales were $35,686M in 2025 [1].",
    "Net sales were $35.686 billion in 2025 [1].",
    "Net sales were 35,686 in 2025, in millions [1].",  # a bare table cell, no $
    "Diluted EPS was $7.46 [1].",                        # per-share amounts are dollars as written
])
def test_amounts_from_a_table_in_millions_pass_with_their_unit(world, answer):
    assert _no_number_warning(_answer_from_table(world, answer))


@pytest.mark.parametrize("answer", [
    "Net sales were $35,686 in 2025 [1].",
    "Net sales fell by $1,319 [1].",
])
def test_a_dollar_amount_that_lost_its_millions_is_caught(world, answer):
    """A real answer said Greater China net sales were "$64,377" at HIGH: the
    table's digits, but read as $64 thousand instead of $64 billion."""
    a = _answer_from_table(world, answer)
    assert not _no_number_warning(a) and a.confidence.band == "low"


# Note 3 as the index holds it: one table row per line. The fake grader reads
# "grade=" from an item's first line only.
EPS_TABLE = "\n".join([
    "Note 3 - Earnings Per Share grade=0.9",
    "(net income in millions and shares in thousands):",
    "2025 | 2024 | 2023",
    "Net income | $112,010 | $93,736 | $96,995",
    "Weighted-average diluted shares | 15,004,697 | 15,408,095 | 15,812,547",
    "Diluted earnings per share | $7.46 | $6.08 | $6.13",
])


def _answer_from_eps_table(world, answer: str):
    p = passages("a1", 0.9, 0.9)
    p[0].chunk.text = EPS_TABLE
    world([p], groq=FakeGroq(answer={"answerable": True, "citations": ["[1]"], "answer": answer}))
    return gen.answer("Why did diluted EPS grow faster than net income?")


def test_share_counts_in_thousands_and_changes_within_a_row_are_allowed(world):
    """Eval Q18's real answer was right but marked LOW: the table states
    shares "in thousands", and the percentages are changes along its rows."""
    a = _answer_from_eps_table(world, (
        "Net income increased 19.5% from $93,736 million to $112,010 million [1]. Diluted shares declined "
        "from 15,408,095 thousand to 15,004,697 thousand, a drop of about 2.6% [1]. Diluted EPS rose 22.7% "
        "from $6.08 per share to $7.46 per share [1]."))
    assert _no_number_warning(a)


@pytest.mark.parametrize("answer", [
    "Net income increased 25.0% [1].",                     # matches no change in any row
    "Diluted EPS rose from $6.08 to $8.12 per share [1].",  # not in the table
])
def test_a_percentage_or_figure_not_in_the_table_is_still_caught(world, answer):
    assert not _no_number_warning(_answer_from_eps_table(world, answer))


def test_a_database_figure_without_its_millions_is_caught(world):
    world([passages("a1", 0.1)], facts=[fact("Net income", 112_010e6)],
          groq=FakeGroq(fact_grades={"Net income": 0.9}, answer={
              "answerable": True, "citations": ["[S1]"], "answer": "Net income was $112,010 [S1]."}))
    a = gen.answer("net income FY2025?")
    assert not _no_number_warning(a) and a.confidence.band == "low"


@pytest.mark.parametrize("label, value, table, written", [
    ("Net income", 112_010e6, "income_statement", "$112,010 million"),
    ("Diluted (in dollars per share)", 7.46, "income_statement", "$7.46 per share"),
    ("Cash used in financing activities", -120_686e6, "cash_flow", "-$120,686 million"),
    ("Diluted (in shares)", 15_004_697_000.0, "income_statement", "15,004,697,000 shares"),
    ("Odd amount", 1_234_567.0, "balance_sheet", "$1,234,567"),
])
def test_database_figures_are_described_in_millions_like_the_filing(label, value, table, written):
    assert describe_fact(fact(label, value, table)).startswith(f"{label}, fiscal year 2025: {written} (from")


@pytest.mark.parametrize("question, labels", [
    # A measure limited to one product or region isn't in the database.
    ("What was Apple's iPhone gross margin in FY2025?", []),
    ("What was iPhone's operating income in FY2025?", []),
    ("What was the gross margin for the iPhone in FY2025?", []),
    ("iPhone gross margin", []),  # the fuzzy pass must not fetch it instead
    ("What was Products gross margin in FY2025?", []),
    ("What was Greater China net sales in FY2025?", []),
    # Elsewhere in the question, the total may be what's wanted: it's fetched,
    # and Layer 5 keeps an answer resting on it below HIGH (tests below).
    ("How much gross margin did the iPhone make in FY2025?", ["Gross margin", "iPhone"]),
    ("How do tariffs in Europe affect Apple's gross margin in FY2025?", ["Gross margin"]),
    # A product's row is its net sales: only the company-wide total goes.
    ("What were Services net sales in FY2025?", ["Services"]),
    ("What were net sales of the iPhone in FY2025?", ["iPhone"]),
    # Asked about separately, both figures are wanted.
    ("What were iPhone revenue and gross margin in FY2025?", ["iPhone", "Gross margin"]),
    ("Did net income grow faster than iPhone revenue from FY2024 to FY2025?", ["Net income", "iPhone"]),
    ("How did iPhone sales compare with net income in FY2025?", ["iPhone", "Net income"]),
    ("What was iPhone revenue as a share of net sales in FY2025?", ["iPhone", "Net sales"]),
    ("What was Apple's gross margin in FY2025?", ["Gross margin"]),
])
def test_a_measure_limited_to_one_product_or_region_fetches_no_company_figure(question, labels):
    found = list(dict.fromkeys(m.label for m in match_structured_query(question)))
    assert found == labels


@pytest.mark.parametrize("question, named", [
    ("How do tariffs in Europe affect Apple's gross margin in FY2025?", ["europe"]),
    ("How much gross margin did the iPhone make in FY2025?", ["iphone"]),
    ("Why did Services net sales increase in FY2025?", []),  # the product's own net sales
    ("Did Apple's Services revenue grow faster than its iPhone revenue?", []),
])
def test_products_and_regions_a_question_is_about(question, named):
    assert segments_mentioned(question) == named


def _company_wide_answer(world, question: str, facts: list[dict], answer: str, citations: list[str]):
    world([passages("a1", 0.1)], facts=facts,
          groq=FakeGroq(fact_grades={"Gross margin": 0.9, "iPhone": 0.9, "Net sales": 0.9},
                        answer={"answerable": True, "citations": citations, "answer": answer}))
    return gen.answer(question)


def test_a_company_wide_figure_for_a_question_about_a_region_is_not_high(world):
    a = _company_wide_answer(
        world, "How do tariffs in Europe affect Apple's gross margin in FY2025?",
        [fact("Gross margin", 195_201e6)], "Gross margin was $195,201 million [S1].", ["[S1]"])
    assert a.answered and a.confidence.band != "high"
    assert any("question is about europe" in n and "company-wide" in n for n in a.confidence.notes)


def test_a_company_wide_figure_beside_the_products_own_is_fine(world):
    a = _company_wide_answer(
        world, "What was iPhone's share of net sales in FY2025?",
        [fact("iPhone", 209_586e6), fact("Net sales", 416_161e6)],
        "iPhone revenue was $209,586 million [S1] of $416,161 million net sales [S2].", ["[S1]", "[S2]"])
    assert a.confidence.band == "high"
    assert not any("company-wide" in n for n in a.confidence.notes)


@pytest.mark.parametrize("question, labels", [
    ("What was iPhone's revenue in FY2025?", ["iPhone"]),
    ("What was iPhone’s revenue in FY2025?", ["iPhone"]),
    ("How much did iPhones bring in for FY2025?", ["iPhone"]),
    ("What was Mac's revenue in FY2025?", ["Mac"]),
    ("How much did Apple spend on stock buybacks in FY2025?", ["Repurchases of common stock"]),
    ("What macroeconomic risks does Apple disclose?", []),  # still whole words only
])
def test_possessives_and_plurals_match_their_figure(question, labels):
    """"iPhone's" becomes "iphones" once apostrophes are removed, which the
    whole-word match for "iphone" used to miss - so no figure was fetched."""
    found = list(dict.fromkeys(m.label for m in match_structured_query(question)))
    assert found == labels


@pytest.mark.parametrize("question, labels", [
    ("How many diluted shares did Apple have in FY2025?", ["Diluted (in shares)"]),
    ("What was Apple's diluted share count in FY2025?", ["Diluted (in shares)"]),
    ("What was Apple's basic share count in FY2025?", ["Basic (in shares)"]),
    ("What was Apple's diluted EPS in FY2025?", ["Diluted (in dollars per share)"]),
    ("How did stock buybacks affect diluted shareholders in FY2025?", ["Repurchases of common stock"]),
])
def test_share_counts_are_matched_apart_from_eps(question, labels):
    found = list(dict.fromkeys(m.label for m in match_structured_query(question)))
    assert found == labels


@pytest.mark.parametrize("answer, ok", [
    ("Apple had 15,004,697,000 diluted shares [S1].", True),
    ("Apple had about 15.00 billion diluted shares [S1].", True),
    ("Apple had 15,004,697 thousand diluted shares [S1].", True),
    ("Apple had 15,004,697 diluted shares [S1].", False),  # the table's figure without "thousand"
    ("Apple had 15,812,547 thousand diluted shares [S1].", False),  # FY2023's figure
])
def test_share_count_answers_are_checked_with_their_unit(world, answer, ok):
    shares = {**fact("Diluted (in shares)", 15_004_697_000.0), "label": "Diluted (in shares)"}
    world([passages("a1", 0.1)], facts=[shares],
          groq=FakeGroq(fact_grades={"Diluted (in shares)": 0.9},
                        answer={"answerable": True, "citations": ["[S1]"], "answer": answer}))
    a = gen.answer("How many diluted shares did Apple have in FY2025?")
    assert _no_number_warning(a) is ok


def test_an_answer_in_millions_matches_the_stored_value(world):
    world([passages("a1", 0.1)], facts=[fact("Net income", 112_010e6)],
          groq=FakeGroq(fact_grades={"Net income": 0.9}, answer={
              "answerable": True, "citations": ["[S1]"], "answer": "Net income was $112,010 million [S1]."}))
    a = gen.answer("net income FY2025?")
    assert _no_number_warning(a) and a.confidence.band == "high"


def test_a_wrong_number_is_still_caught(world):
    world([passages("a1", 0.1)], facts=[fact("Net income", 112_010e6)],
          groq=FakeGroq(fact_grades={"Net income": 0.9}, answer={
              "answerable": True, "citations": ["[S1]"], "answer": "Net income was $11.2 billion [S1]."}))
    a = gen.answer("net income FY2025?")
    assert not _no_number_warning(a) and a.confidence.band == "low"


def test_the_answer_prompt_names_the_section_when_a_passage_has_no_subsection():
    """The grader sees "Item 1A / Risk Factors" (see _where); the answer model
    must see the same location, not "Item 1A / " with nothing after the slash."""
    retrieved = passages("a1", 0.9)  # item "Item 1A", section "Risk Factors", no subsection
    evidence, _ = gen._build_evidence([GradedChunk(retrieved=retrieved[0], relevance=0.9)], [])
    assert "[1] (Item 1A / Risk Factors)" in evidence


def test_a_failure_says_what_went_wrong_not_just_its_type():
    """A crash in search (a model that wouldn't load, a locked index) is also
    reported as "Model call failed". Showing only "OSError" hid the cause."""
    a = gen._model_failed("q", OSError("[WinError 1455] The paging file is too small\nsecond line"))
    assert a.text.startswith("Model call failed (OSError: [WinError 1455] The paging file is too small)")
    assert "second line" not in a.text
    assert gen._model_failed("q", RuntimeError()).text.startswith("Model call failed (RuntimeError).")
# --- confidence of answers resting on database figures ----------------------------

W = get_config().confidence.weights


def test_a_single_well_graded_figure_keeps_its_score(world):
    world([passages("a1", 0.1)], facts=[fact("Net income", 112_010e6)],
          groq=FakeGroq(fact_grades={"Net income": 0.9}, answer={
              "answerable": True, "citations": ["[S1]"], "answer": "Net income was $112,010,000,000 [S1]."}))
    a = gen.answer("net income FY2025?")
    assert a.confidence.score == 0.9 and a.confidence.band == "high"


def test_a_borderline_figure_drops_to_medium(world):
    # The uncited 0.6 passage only gets Layer 4 past "2 relevant items".
    world([passages("a1", 0.6)], facts=[fact("Net income", 112_010e6)],
          groq=FakeGroq(fact_grades={"Net income": 0.55}, answer={
              "answerable": True, "citations": ["[S1]"], "answer": "Net income was $112,010,000,000 [S1]."}))
    a = gen.answer("net income FY2025?")
    assert a.confidence.score == pytest.approx(0.9 * 0.55 / CFG.strong_chunk_threshold, abs=0.001)
    assert a.confidence.band == "medium"
    assert any("graded 0.55" in n for n in a.confidence.notes)


def test_a_passage_that_restates_the_figure_does_not_lower_it(world):
    """The income statement table writes net income as "112,010" (millions):
    the same number, so citing it alongside [S1] changes nothing."""
    p = passages("a1", 0.9)
    p[0].chunk.text = "CONSOLIDATED STATEMENTS OF OPERATIONS (in millions) Net income 112,010 93,736 grade=0.9"
    world([p], facts=[fact("Net income", 112_010e6)],
          groq=FakeGroq(fact_grades={"Net income": 0.9}, answer={
              "answerable": True, "citations": ["[S1]", "[1]"],
              "answer": "Net income was $112,010,000,000 [S1][1]."}))
    a = gen.answer("net income FY2025?")
    assert a.confidence.score == 0.9 and a.confidence.band == "high"
    assert not any("blended" in n for n in a.confidence.notes)


def test_a_passage_that_adds_a_reason_is_blended_in(world):
    """The passage below was found by one search only and has no reranker
    score, so its search confidence is just the first-try part of the formula."""
    p = passages("a1", 0.9)
    p[0].chunk.text = "Services net sales grew due to higher advertising and cloud services. grade=0.9"
    world([p], facts=[fact("Services", 109_158e6)],
          groq=FakeGroq(fact_grades={"Services": 0.9}, answer={
              "answerable": True, "citations": ["[S1]", "[1]"],
              "answer": "Services net sales were $109,158,000,000 [S1], driven by advertising and cloud [1]."}))
    a = gen.answer("Why did Services grow?")
    assert a.confidence.score == pytest.approx((0.9 + W.loop_cleanliness * 1.0) / 2, abs=0.001)
    assert a.confidence.band != "high"
    assert any("1 cited passage(s) carry part of the answer" in n for n in a.confidence.notes)


# --- a second search when the answer step finds nothing answers it ---------------
# Eval Q20: "what caused the FY2024 net income dip?" finds the net income lines;
# the answer step rightly refuses; the cause (a tax charge) sits where the
# question's words never reach - but a search for one of its parts does.

REFUSAL = {"answerable": False, "answer": "Nothing here explains the dip.", "citations": []}
TAX_ANSWER = {"answerable": True, "answer": "A one-time tax charge lowered FY2024 net income [3].",
              "citations": ["[3]"]}
PARTS = ["Provision for income taxes in FY2024", "One-time charges in FY2024"]


def test_a_refused_answer_gets_one_more_search_with_the_question_split(world):
    found, cause = passages("a1", 0.9, 0.6), passages("tax", 0.9)
    groq = FakeGroq(answer=[REFUSAL, TAX_ANSWER], parts=PARTS)
    world([found, cause, cause], groq=groq)  # the question's search, then one per part
    events = []
    a = gen.answer("What caused the FY2024 dip?", on_progress=lambda s, d: events.append((s, d)))

    assert a.answered and [c.chunk_id for c in a.citations] == ["tax-0"]
    assert "split" in stages(events)
    assert a.outcome.attempts[-1].strategy == "decompose"
    assert any("second search" in n for n in a.confidence.notes)
    # Graded against the question as asked, with its parts alongside: without
    # them a real grader scored the tax table 0.0, "no cause of dip".
    second_grading = groq.prompts["chunk_grades"][-1]
    assert "What caused the FY2024 dip?" in second_grading
    assert all(f"- {p}" in second_grading for p in PARTS)
    assert "- " + PARTS[0] not in groq.prompts["chunk_grades"][0], "the first grading has no parts"


def test_a_question_that_cannot_be_split_keeps_its_refusal(world):
    groq = FakeGroq(answer=[REFUSAL, TAX_ANSWER], parts=[])
    world([passages("a1", 0.9, 0.6)], groq=groq)
    a = gen.answer("What caused the FY2024 dip?")
    assert not a.answered and a.text == REFUSAL["answer"]
    assert len(groq.prompts["grounded_answer"]) == 1


def test_the_second_search_happens_once(world):
    groq = FakeGroq(answer=[REFUSAL, REFUSAL, TAX_ANSWER], parts=PARTS)
    world([passages("a1", 0.9, 0.6), passages("tax", 0.9)], groq=groq)
    a = gen.answer("What caused the FY2024 dip?")
    assert not a.answered
    assert len(groq.prompts["question_parts"]) == 1 and len(groq.prompts["grounded_answer"]) == 2


def test_a_question_layer_4_refuses_also_gets_the_second_search(world):
    """The same question was refused by Layer 4 on one real run and by the
    answer step on another, depending on how the net income lines graded."""
    # Nothing from the first search passed, so the part's passages are [1] and [2].
    groq = FakeGroq(answer={**TAX_ANSWER, "answer": "A one-time tax charge [1].", "citations": ["[1]"]},
                    parts=PARTS)
    weak, cause = passages("a1", 0.05), passages("tax", 0.9, 0.9)
    world([weak, cause, cause], groq=groq)
    a = gen.answer("What caused the FY2024 dip?")
    assert len(groq.prompts["question_parts"]) == 1
    assert a.answered and a.outcome.attempts[-1].strategy == "decompose"


def test_a_question_answered_the_first_time_gets_no_second_search(world):
    groq = FakeGroq(parts=PARTS)
    world([passages("a1", 0.9, 0.6)], groq=groq)
    assert gen.answer("question?").answered
    assert "question_parts" not in groq.prompts


def test_a_second_search_that_finds_nothing_keeps_the_refusal(world):
    groq = FakeGroq(answer=TAX_ANSWER, parts=PARTS)
    world([passages("a1", 0.05), passages("x", 0.05)], groq=groq)
    a = gen.answer("What is Apple's projected revenue for FY2027?")
    assert not a.answered and "question_parts" in groq.prompts
    assert "grounded_answer" not in groq.prompts, "nothing passed, so no answer was written"
