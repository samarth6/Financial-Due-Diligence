"""Show one question's answer from two eval runs side by side.

    python eval/compare_runs.py eval/results_round2_before.jsonl eval/results.jsonl cybersecurity

The last argument is any part of the question text (not case-sensitive).
"""

import json
import sys


def load(path: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return {r["question"]: r for r in rows if "question" in r}  # skips the _meta line


before_path, after_path, search = sys.argv[1], sys.argv[2], sys.argv[3].lower()
before, after = load(before_path), load(after_path)

for question in [q for q in after if search in q.lower()] or ["(no question matches)"]:
    print(f"\n=== {question}\n")
    for label, rows in (("BEFORE", before), ("AFTER", after)):
        r = rows.get(question)
        if r is None:
            print(f"{label}: (not in this run)\n")
            continue
        status = "answered" if r["answered"] else "REFUSED"
        print(f"{label}: {status}, band={r['band']}, confidence={r['confidence']}")
        print(r["text"])
        for c in r["citations"]:
            print(f"  {c}")
        print()
