"""Run the eval set through the full pipeline (Layers 3-5) and score it.

    python eval/run_eval.py                       # all 26 questions
    python eval/run_eval.py numeric_lookup        # one type only
    python eval/run_eval.py numeric_lookup no_answer

Scoring:
- numeric_lookup: automatic - the expected number must appear in the answer
  (as a full number, "X million" or "X billion" all count).
- no_answer: automatic - correct only if the system refused to answer.
- narrative_comprehension / cross_referencing / trend: graded by hand. These
  are free text; the script writes eval/review.md with expected and actual
  answers side by side.

Each result is saved to eval/results.jsonl the moment it finishes, and
questions already in that file are skipped. Groq's free tier caps tokens per
day, so a full run may stop partway - just run it again later and it picks
up where it left off. Delete results.jsonl to start over.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dd.generation import answer

EVAL_DIR = Path(__file__).parent
EVAL_SET = EVAL_DIR / "eval_set.json"
RESULTS = EVAL_DIR / "results.jsonl"
REVIEW = EVAL_DIR / "review.md"

AUTO_TYPES = {"numeric_lookup", "no_answer"}

# --- scoring ---------------------------------------------------------------

_MARKER_RE = re.compile(r"\[S?\d+\]")
_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(million|billion)?", re.IGNORECASE)
_SCALE = {"million": 1e6, "billion": 1e9}


def _numbers_in(text: str) -> list[float]:
    """Every number in the text, with 'million'/'billion' applied.
    Citation markers are removed first so [1] isn't read as the number 1."""
    text = _MARKER_RE.sub(" ", text)
    out = []
    for digits, unit in _NUMBER_RE.findall(text):
        value = float(digits.replace(",", ""))
        out.append(value * _SCALE.get(unit.lower(), 1) if unit else value)
    return out


def _expected_number(expected: str) -> float:
    """The first number in the expected answer: '$112,010,000,000 (~$112.0
    billion)' -> 112010000000.0. The part in brackets is just a restatement."""
    return _numbers_in(expected)[0]


def score(item: dict, result: dict) -> bool | None:
    """True/False for auto-graded types, None for ones graded by hand."""
    kind = item["type"]
    if kind == "no_answer":
        return not result["answered"]
    if kind == "numeric_lookup":
        if not result["answered"]:
            return False
        target = _expected_number(item["expected_answer"])
        # 0.1% tolerance: "$112.0 billion" is a rounded 112,010,000,000.
        return any(abs(n - target) <= 0.001 * abs(target) for n in _numbers_in(result["text"]))
    return None


# --- running ---------------------------------------------------------------


def _load_done() -> dict[str, dict]:
    if not RESULTS.exists():
        return {}
    rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["question"]: r for r in rows}


def run(types: set[str] | None) -> list[dict]:
    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if types:
        items = [i for i in items if i["type"] in types]
    done = _load_done()

    for n, item in enumerate(items, 1):
        q = item["question"]
        if q in done:
            print(f"[{n}/{len(items)}] skip (already done): {q[:60]}")
            continue
        print(f"[{n}/{len(items)}] {item['type']}: {q[:60]}", flush=True)
        try:
            a = answer(q)
        except Exception as exc:  # noqa: BLE001 - one failure shouldn't end the run
            # Not saved, so the next run retries it. Usually a rate limit.
            print(f"    ERROR (will retry next run): {type(exc).__name__}: {exc}")
            continue

        result = {
            "question": q,
            "type": item["type"],
            "expected": item["expected_answer"],
            "answered": a.answered,
            "text": a.text,
            "citations": [f"{c.marker} {c.source}" for c in a.citations],
            "confidence": a.confidence.score if a.confidence else None,
            "band": a.confidence.band if a.confidence else None,
            "attempts": len(a.outcome.attempts) if a.outcome else 0,
            "elapsed_s": a.elapsed_s,
        }
        result["correct"] = score(item, result)
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        done[q] = result
        mark = {True: "CORRECT", False: "WRONG", None: "(grade by hand)"}[result["correct"]]
        print(f"    {mark}  band={result['band']}  {result['elapsed_s']}s")

    return [done[i["question"]] for i in items if i["question"] in done]


# --- reporting -------------------------------------------------------------


def summarize(results: list[dict]) -> None:
    print("\n=== by question type ===")
    for kind in sorted({r["type"] for r in results}):
        rows = [r for r in results if r["type"] == kind]
        if kind in AUTO_TYPES:
            ok = sum(1 for r in rows if r["correct"])
            print(f"  {kind:<26} {ok}/{len(rows)} correct")
        else:
            refused = sum(1 for r in rows if not r["answered"])
            print(f"  {kind:<26} {len(rows)} to grade by hand ({refused} refused)")

    # Does "high" confidence actually mean "usually right"? Only answerable
    # auto-graded questions can say - a no_answer refusal has no band.
    graded = [r for r in results if r["correct"] is not None and r["band"]]
    if graded:
        print("\n=== confidence vs. correctness (auto-graded answers only) ===")
        for band in ("high", "medium", "low"):
            rows = [r for r in graded if r["band"] == band]
            if rows:
                ok = sum(1 for r in rows if r["correct"])
                print(f"  {band:<7} {ok}/{len(rows)} correct")

    times = [r["elapsed_s"] for r in results]
    if times:
        print(f"\naverage time: {sum(times) / len(times):.1f}s per question")


def write_review(results: list[dict]) -> None:
    manual = [r for r in results if r["type"] not in AUTO_TYPES]
    if not manual:
        return
    lines = ["# Answers to grade by hand", "",
             "For each: is the answer correct and supported by its sources? "
             "Write correct / partly / wrong next to Verdict.", ""]
    for n, r in enumerate(manual, 1):
        lines += [
            f"## {n}. [{r['type']}] {r['question']}", "",
            f"**Expected:** {r['expected']}", "",
            f"**System ({'answered' if r['answered'] else 'REFUSED'}, band={r['band']}):** {r['text']}", "",
            *[f"- {c}" for c in r["citations"]], "",
            "**Verdict:** ", "",
        ]
    REVIEW.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {REVIEW} - {len(manual)} answers to grade by hand")


if __name__ == "__main__":
    types = set(sys.argv[1:]) or None
    results = run(types)
    summarize(results)
    write_review(results)