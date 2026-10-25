# Graded answers — eval run of 2026-10-21, after the "one strong chunk" rule

Verdicts: **correct** (all key points, nothing wrong) · **partly** (right, but misses a key point) · **wrong** (states something false) · **refused** (said it couldn't answer, but the filing does answer it).

## Summary

| Type | Correct | Partly | Wrong | Refused |
|---|---|---|---|---|
| numeric_lookup (auto) | 8 | 0 | 0 | 0 |
| no_answer (auto) | 5 | 0 | 0 | 0 |
| narrative_comprehension | 6 | 1 | 0 | 0 |
| cross_referencing | 2 | 0 | 0 | 3 |
| trend | 3 | 1 | 0 | 0 |
| **All 29 — this run** | **24** | **2** | **0** | **3** |

Progress across the three runs:

| Run | Correct | Partly | Wrong | Refused |
|---|---|---|---|---|
| 1. Baseline | 19 | 3 | 0 | 7 |
| 2. + multi-number lookup | 23 | 2 | 0 | 4 |
| 3. + one-strong-chunk rule | 24 | 2 | 0 | 3 |

- **Still zero wrong answers** across all three runs.
- **The 5 trick questions are still refused**, so the looser success rule didn't make the system answer what it shouldn't.
- **HIGH confidence: 17 of 18 correct, 1 partly** (#13). MEDIUM and LOW are still mixed; the only LOW answer (#2) is correct.

## 1. [narrative] Tariffs — **partly** (medium)
Right effects; still no countries, no Q2 FY2025 start, no retaliatory tariffs.

## 2. [narrative] Cybersecurity — **correct** (low)
Regular attacks, measures may be insufficient, harms. Misses "higher-value target due to high profile". Correct answer scored LOW, for the third run in a row.

## 3. [narrative] Natural disasters — **correct** (high) — *fixed*
Matches the expected answer, including climate change. Succeeded on attempt 1 via the one-strong-chunk rule.

## 4. [narrative] Wearables decrease — **correct** (high)

## 5. [narrative] Services increase — **correct** (high)

## 6. [narrative] Acquisitions — **correct** (high)
All the expected risks: distraction, liabilities, regulatory challenges, inadequate returns, impairment, approvals.

## 7. [narrative] Legal proceedings — **correct** (medium)
Ordinary course, increasing over time. Doesn't say "some unresolved". Nothing wrong.

## 8. [cross-ref] Assets = Liabilities + Equity — **correct** (high)
359,241,000,000 = 285,508,000,000 + 73,733,000,000. Checked: exact.

## 9. [cross-ref] Cash flow reconciliation — **refused**
The question writes "(operating, investing, financing)", not "operating activities", so the lookup finds nothing; "change in cash" isn't a label either.

## 10. [cross-ref] EPS grew faster than net income — **refused**
Has EPS and net income for both years; says share count and buybacks are missing. Correct reason for refusing.

## 11. [cross-ref] Products gross margin — **correct** (high)
Matches filing lines 730 and 732. Adds "mix shift toward lower-margin products", which follows from the filing's wording. The eval set's expected answer is still wrong ("higher sales volume").

## 12. [cross-ref] FY2024 net income dip — **refused**
The reason (State Aid tax charge) is in the notes; text search misses it.

## 13. [trend] Services vs iPhone growth — **partly** (high)
Now cites the four database figures ([S1]–[S4]), which is why it moved to HIGH. But the answer text still gives no numbers or growth rates (13.5% vs 4.2%). **The one HIGH answer that isn't fully correct.**

## 14. [trend] Net income FY2023–FY2025 — **correct** (high)
96,995 → 93,736 → 112,010 (millions), down then up.

## 15. [trend] Diluted EPS FY2023–FY2025 — **correct** (high)
$6.13 → $6.08 → $7.46.

## 16. [trend] Total liabilities FY2024 → FY2025 — **correct** (high)
Fell by $22,522,000,000, from $308,030,000,000 to $285,508,000,000. Checked: exact.
