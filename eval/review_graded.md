# Graded answers — eval run of 2026-10-20, after multi-number lookup

Verdicts: **correct** (all key points, nothing wrong) · **partly** (right, but misses a key point) · **wrong** (states something false) · **refused** (said it couldn't answer, but the filing does answer it).

## Summary

| Type | Correct | Partly | Wrong | Refused |
|---|---|---|---|---|
| numeric_lookup (auto) | 8 | 0 | 0 | 0 |
| no_answer (auto) | 5 | 0 | 0 | 0 |
| narrative_comprehension | 5 | 1 | 0 | 1 |
| cross_referencing | 2 | 0 | 0 | 3 |
| trend | 3 | 1 | 0 | 0 |
| **All 29 — after** | **23** | **2** | **0** | **4** |
| All 29 — before | 19 | 3 | 0 | 7 |

- **4 answers rescued by the fix, all correct:** #8 (balance sheet equation), #14, #15, #16. Arithmetic in #8 and #16 checked by hand.
- **Still zero wrong answers.**
- **Two changes NOT caused by the fix** (run-to-run variance): #6 went partly → correct; #3 went correct → refused.
- **HIGH confidence: 16 of 16 correct** across both auto- and hand-graded answers in this run. MEDIUM and LOW are mixed (the only LOW, #2, is correct).

## 1. [narrative] Tariffs — **partly** (medium)
Same as last run: right effects, but no countries, no Q2 FY2025 start, no retaliatory tariffs.

## 2. [narrative] Cybersecurity — **correct** (low)
Regular attacks, measures may be insufficient, harms. Misses "higher-value target due to high profile". Correct answer scored LOW again.

## 3. [narrative] Natural disasters — **refused**
"Only 1 relevant chunk after 3 attempts." Was correct last run. Cause: the "need 2 relevant chunks" rule; the answer sits in one paragraph, so success depends on a second, weaker chunk crossing 0.5. A rerun with `dd heal` succeeded on attempt 2.

## 4. [narrative] Wearables decrease — **correct** (high)

## 5. [narrative] Services increase — **correct** (high)

## 6. [narrative] Acquisitions — **correct** (high)
Now includes management distraction, higher liabilities, inadequate returns, impairment. Improved since last run, but by variance, not by the code change.

## 7. [narrative] Legal proceedings — **correct** (medium)
Includes "not yet fully resolved" and "increased over time". Long, nothing wrong.

## 8. [cross-ref] Assets = Liabilities + Equity — **correct** (high) — *rescued*
359,241,000,000 = 285,508,000,000 + 73,733,000,000. Checked: exact.

## 9. [cross-ref] Cash flow reconciliation — **refused**
No "change in cash" label in the lookup list.

## 10. [cross-ref] EPS grew faster than net income — **refused**
Better refusal than last time: it now has EPS and net income for both years, and says what's missing (share count, buybacks).

## 11. [cross-ref] Products gross margin — **correct** (high)
Matches filing lines 730 and 732. The expected answer is wrong ("higher sales volume" isn't in the filing).

## 12. [cross-ref] FY2024 net income dip — **refused**
Reason (State Aid tax charge) is in the notes; text search misses it.

## 13. [trend] Services vs iPhone growth — **partly** (medium)
Right conclusion, still no numbers. It cited a text chunk ([1]) instead of the database facts, which is worth investigating.

## 14. [trend] Net income FY2023–FY2025 — **correct** (high) — *rescued*
96,995 → 93,736 → 112,010 (millions), "decline then increase". Matches.

## 15. [trend] Diluted EPS FY2023–FY2025 — **correct** (high) — *rescued*
$6.13 → $6.08 → $7.46. Matches. Lists the values without describing the pattern in words.

## 16. [trend] Total liabilities FY2024 → FY2025 — **correct** (high) — *rescued*
Decreased by $22,522,000,000. Checked: 308,030 − 285,508 = 22,522 (millions). Doesn't give the % (~7.3%).
