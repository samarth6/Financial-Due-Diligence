"""Run the eval set through the full pipeline (Layers 3-5) and score it.

    python eval/run_eval.py                       # every question in eval_set.json (29)
    python eval/run_eval.py numeric_lookup        # one type only
    python eval/run_eval.py numeric_lookup no_answer
    python eval/run_eval.py --repeat 3            # each question 3 times

Scoring:
- numeric_lookup: automatic - the expected number must appear in a sentence
  of the answer that carries a citation, so a number the answer only
  mentions in passing doesn't count. "X million", "X billion", and bare
  numbers in an answer that says "in millions" all count. Bracketed
  negatives like (1,234) are compared by size.
- no_answer: automatic - correct only if the system refused to answer.
- narrative_comprehension / cross_referencing / trend: graded by hand. These
  are free text; the script writes eval/review.md with expected and actual
  answers side by side.

What produced the results: the first line of eval/results.jsonl records the
git commit, a fingerprint of any uncommitted changes to src/ and config.yaml,
fingerprints of config.yaml and eval_set.json, the model names and the date.
A run only resumes if all of those still match - otherwise it stops, so old
and new answers never get mixed in one file.

Resuming: each result is saved the moment it finishes, and questions already
in the file are skipped. Groq's free tier caps tokens per day, so a full run
may stop partway - run it again later and it picks up where it left off.

--repeat N: the model doesn't give the same answer every time. With N runs,
the summary shows how often each question passed, and marks a question
"unstable" when it passed some runs and failed others. A 3x run is ~87
questions, more than one day of Groq's free tier - resume covers that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dd.generation import answer

EVAL_DIR = Path(__file__).parent
ROOT = EVAL_DIR.parent
EVAL_SET = EVAL_DIR / "eval_set.json"
CONFIG = ROOT / "config.yaml"
RESULTS = EVAL_DIR / "results.jsonl"
REVIEW = EVAL_DIR / "review.md"

AUTO_TYPES = {"numeric_lookup", "no_answer"}

# --- scoring ---------------------------------------------------------------

_MARKER_RE = re.compile(r"\[S?\d+\]")
_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(million|billion)?", re.IGNORECASE)
_SCALE = {"million": 1e6, "billion": 1e9}
# A new sentence starts after . ! or ? and a space - but not before a
# citation marker, so "...in 2025. [S1]" keeps its marker.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\[)")


def _numbers_in(text: str, in_millions: bool = False) -> list[float]:
    """Every number in the text, with 'million'/'billion' applied.
    Citation markers are removed first so [1] isn't read as the number 1.
    If the answer says "in millions", a bare number may be in millions, so
    both readings are kept: 112,010 -> 112010 and 112010000000."""
    text = _MARKER_RE.sub(" ", text)
    out = []
    for digits, unit in _NUMBER_RE.findall(text):
        value = float(digits.replace(",", ""))
        if unit:
            out.append(value * _SCALE[unit.lower()])
        else:
            out.append(value)
            if in_millions:
                out.append(value * 1e6)
    return out


def _cited_numbers(text: str) -> list[float]:
    """Numbers from sentences that carry a citation marker - the ones the
    answer actually backs with a source."""
    in_millions = "in millions" in text.lower()
    cited = [s for s in _SENTENCE_SPLIT_RE.split(text) if _MARKER_RE.search(s)]
    return [n for s in cited for n in _numbers_in(s, in_millions)]


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
        target = abs(_expected_number(item["expected_answer"]))
        # 0.1% tolerance: "$112.0 billion" is a rounded 112,010,000,000.
        # abs(): a bracketed negative "(1,234)" is read as 1,234.
        return any(abs(n - target) <= 0.001 * target for n in _cited_numbers(result["text"]))
    return None


# --- what produced these results -------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def _model_names(node, path: str = "") -> dict[str, str]:
    """Every config.yaml setting whose name contains 'model', e.g.
    generation.utility_model -> openai/gpt-oss-20b."""
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(_model_names(value, f"{path}{key}."))
    elif "model" in path:
        out[path.rstrip(".")] = str(node)
    return out


def current_meta() -> dict:
    config_text = CONFIG.read_text(encoding="utf-8")
    # Uncommitted edits change the code without changing the commit, so they
    # get their own fingerprint. "" = clean.
    diff = _git("diff", "HEAD", "--", "src", "config.yaml")
    return {
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "uncommitted_changes": _sha(diff) if diff else "",
        "config_sha": _sha(config_text),
        "eval_set_sha": _sha(EVAL_SET.read_text(encoding="utf-8")),
        "models": _model_names(yaml.safe_load(config_text)),
    }


# --- running ---------------------------------------------------------------


def _load_done(meta: dict) -> dict[tuple[str, int], dict]:
    """Results already in results.jsonl, keyed by (question, run number).
    Refuses to continue a file made by different code, config or questions."""
    if not RESULTS.exists():
        RESULTS.write_text(
            json.dumps({"_meta": {**meta, "started": _now()}}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {}

    lines = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    saved = lines[0].get("_meta") if lines else None
    if saved is None:
        sys.exit(
            f"{RESULTS} has no record of what produced it (made by an older version "
            "of this script). Rename it, e.g. results_old.jsonl, and run again."
        )
    changed = [k for k in meta if saved.get(k) != meta[k]]
    if changed:
        sys.exit(
            f"{RESULTS} was made with a different {', '.join(changed)}:\n"
            + "\n".join(f"  {k}: then {saved.get(k)!r}, now {meta[k]!r}" for k in changed)
            + "\nMixing the two would make the results meaningless. Rename the file "
            "(e.g. results_<what changed>.jsonl) and run again."
        )
    return {(r["question"], r.get("run", 1)): r for r in lines[1:]}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(types: set[str] | None, repeat: int) -> list[dict]:
    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if types:
        items = [i for i in items if i["type"] in types]
    meta = current_meta()
    if meta["uncommitted_changes"]:
        print("note: src/ or config.yaml has uncommitted changes - commit first so "
              "these results can be traced to a commit.\n")
    done = _load_done(meta)

    total = len(items) * repeat
    n = 0
    for run_no in range(1, repeat + 1):
        for item in items:
            n += 1
            q = item["question"]
            if (q, run_no) in done:
                continue
            print(f"[{n}/{total}] run {run_no} {item['type']}: {q[:60]}", flush=True)
            try:
                a = answer(q)
            except Exception as exc:  # noqa: BLE001 - one failure shouldn't end the run
                # Not saved, so the next run retries it. Usually a rate limit.
                print(f"    ERROR (will retry next run): {type(exc).__name__}: {exc}")
                continue

            result = {
                "question": q,
                "run": run_no,
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
            done[(q, run_no)] = result
            mark = {True: "CORRECT", False: "WRONG", None: "(grade by hand)"}[result["correct"]]
            print(f"    {mark}  band={result['band']}  {result['elapsed_s']}s")

    return [done[(i["question"], r)] for r in range(1, repeat + 1) for i in items
            if (i["question"], r) in done]


# --- reporting -------------------------------------------------------------


def _by_question(results: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in results:
        out.setdefault(r["question"], []).append(r)
    return out


def summarize(results: list[dict], repeat: int) -> None:
    groups = _by_question(results)

    print("\n=== by question type ===")
    for kind in sorted({r["type"] for r in results}):
        qs = {q: rows for q, rows in groups.items() if rows[0]["type"] == kind}
        rows = [r for rs in qs.values() for r in rs]
        if kind in AUTO_TYPES:
            ok = sum(1 for r in rows if r["correct"])
            always = sum(1 for rs in qs.values() if all(r["correct"] for r in rs))
            line = f"  {kind:<26} {ok}/{len(rows)} correct"
            if repeat > 1:
                line += f"  ({always}/{len(qs)} questions correct every run)"
            print(line)
        else:
            refused = sum(1 for r in rows if not r["answered"])
            print(f"  {kind:<26} {len(rows)} to grade by hand ({refused} refused)")

    if repeat > 1:
        # Unstable = the same question gave a different outcome across runs:
        # passed some and failed others, or answered some and refused others.
        print("\n=== per question (unstable = different outcome across runs) ===")
        for q, rows in groups.items():
            if rows[0]["type"] in AUTO_TYPES:
                k = sum(1 for r in rows if r["correct"])
                label, flips = f"{k}/{len(rows)} correct", 0 < k < len(rows)
            else:
                k = sum(1 for r in rows if r["answered"])
                label, flips = f"{k}/{len(rows)} answered", 0 < k < len(rows)
            print(f"  {'UNSTABLE ' if flips else '         '}{label:<14} {q[:70]}")

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
    groups = {q: rows for q, rows in _by_question(results).items() if rows[0]["type"] not in AUTO_TYPES}
    if not groups:
        return
    lines = ["# Answers to grade by hand", "",
             "For each: is the answer correct and supported by its sources? "
             "Write correct / partly / wrong next to Verdict.", ""]
    for n, (q, rows) in enumerate(groups.items(), 1):
        lines += [f"## {n}. [{rows[0]['type']}] {q}", "", f"**Expected:** {rows[0]['expected']}", ""]
        for r in rows:
            run = f"run {r.get('run', 1)}, " if len(rows) > 1 else ""
            lines += [
                f"**System ({run}{'answered' if r['answered'] else 'REFUSED'}, band={r['band']}):** {r['text']}", "",
                *[f"- {c}" for c in r["citations"]], "",
                "**Verdict:** ", "",
            ]
    REVIEW.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {REVIEW} - {sum(len(r) for r in groups.values())} answers to grade by hand")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and score the eval set.")
    parser.add_argument("types", nargs="*", help="question types to run (default: all)")
    parser.add_argument("--repeat", type=int, default=1, help="run each question N times")
    args = parser.parse_args()
    results = run(set(args.types) or None, args.repeat)
    summarize(results, args.repeat)
    write_review(results)