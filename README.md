# Financial Due-Diligence RAG

Ask questions about Apple's FY2025 10-K. Every answer carries citations and a
confidence band, and the system refuses when the filing does not contain the
answer. Answers can become approved actions (checklist item, review flag, risk
memo), and a team can work together in a shared web room.

## Layers

| Layer | What it does |
|---|---|
| 1 Ingest | Filing text plus three financial tables (income statement, balance sheet, cash flow) in DuckDB |
| 2 Index | Chunks into Qdrant (meaning search) and BM25 (keyword search) |
| 3 Retrieve | Two searches merged with RRF, then a reranker picks the best 6; plus a database lookup |
| 4 Self-heal | A checker model grades the evidence; reword and retry, or refuse |
| 5 Generate | Cited answer, number check, confidence score |
| 6 Actions | Propose, validate, human approval, execute |
| 7 Team room | FastAPI + WebSocket + React: queue, live steps, group vote |
| Eval | 29 questions with known answers |

## Requirements

- Python 3.14 (what it was tested on)
- Node.js (to build the web page)
- A Groq API key (free tier is enough)
- About 8 GB of RAM (the reranker is large)

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
Copy-Item .env.example .env       # then put your Groq key in .env
cd web; npm install; npm run build; cd ..
```

## Build the data (once)

The raw filing and tables are in `data/raw`.

```
python build_financials_db.py     # run from the repo root; ends with "All checks passed."
dd chunk                          # filing text -> citation-carrying chunks
dd index                          # chunks -> Qdrant + BM25
dd stats                          # check what is on disk
```

## Run

```
dd answer "What was Apple's net income for FY2025?"
dd serve                          # team room at http://localhost:8000
```

Only one process can use the search index at a time. Stop `dd serve` before
running `dd answer`, the eval, or any other `dd` command.

Other commands: `dd retrieve`, `dd heal`, `dd probe`, `dd session`,
`dd act "question" --action checklist|flag|memo`, `dd flags`,
`dd close-flag <id> --note "..."`.

## Tests and eval

```
python -m pytest -q
python eval/run_eval.py [types] [--repeat N]
python eval/compare_runs.py old.jsonl new.jsonl "words from the question"
```

Rename the old `eval/results.jsonl` before a new run. The eval refuses to mix
results from different code.

## Known limitations

(Fill in after the current fixes are merged.)

- Single filing (Apple FY2025 10-K), three years of tables.
- The team room has no login and holds one room on one server.
- Confidence weights are hand-picked and checked only against the eval set.
- Free-tier model limits make answers take 20 to 60 seconds.