"""Layer 6: verified actions.

Layers 3-5 answer a question. This layer lets the system DO something with
an answer - add an item to the due-diligence checklist - but never on the
model's judgement alone:

1. propose  - the model drafts the item in a fixed format
2. validate - code checks it against hard rules; any failure stops it here
3. approve  - a person sees the draft and says yes (the CLI does this step)
4. execute  - only then is it written to the checklist

The draft can only cite sources the Layer 5 answer already cited, so an
action can never rest on evidence the answer didn't have.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import DATA_DIR, Config, get_config
from .models import Answer
from .self_healing import get_groq_client

CHECKLIST_PATH = DATA_DIR / "checklist.json"


class ChecklistItem(BaseModel):
    """The one action this layer supports. Field limits are the first
    check: a draft that doesn't fit this shape never gets as far as the
    rules below."""

    title: str = Field(min_length=5, max_length=120)
    severity: Literal["low", "medium", "high"]
    summary: str = Field(min_length=10, max_length=600)
    citations: list[str] = Field(min_length=1)  # markers from the answer, e.g. "[1]", "[S1]"

    @field_validator("citations")
    @classmethod
    def _bracket_markers(cls, markers: list[str]) -> list[str]:
        """The model sometimes writes "1" or "S1" instead of "[1]" / "[S1]".
        Same source, different spelling - normalise it, so the check in
        validate() only rejects markers that point to nothing."""
        out = []
        for m in markers:
            m = m.strip()
            out.append(m if m.startswith("[") else f"[{m}]")
        return out


# --- Step 1: propose ----------------------------------------------------------

_PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "severity", "summary", "citations"],
}

_PROPOSE_PROMPT = """Turn this answer from a company's 10-K into ONE item for \
a due-diligence checklist.

Question: {question}

Answer:
{answer}

Sources the answer cites:
{sources}

Write:
- title: a short name for the issue (under 100 characters)
- severity: low, medium or high, for how much this matters to an investor
- summary: two or three short sentences, under 400 characters in total, \
  using only facts from the answer
- citations: the markers from the list above that the summary relies on"""


def propose(answer: Answer, cfg: Config | None = None) -> ChecklistItem:
    """Ask the small model for a draft. The JSON schema fixes the shape;
    ChecklistItem then enforces the limits (length, at least one citation)."""
    conf = cfg or get_config()
    sources = "\n".join(f"{c.marker} {c.source}" for c in answer.citations)
    prompt = _PROPOSE_PROMPT.format(question=answer.query, answer=answer.text, sources=sources)

    resp = get_groq_client(conf).chat.completions.create(
        model=conf.generation.utility_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=conf.generation.utility_max_tokens,
        temperature=0.0,
        reasoning_effort=conf.generation.utility_effort,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "checklist_item", "schema": _PROPOSE_SCHEMA},
        },
    )
    return ChecklistItem.model_validate_json(resp.choices[0].message.content)


# --- Step 2: validate ---------------------------------------------------------


def _load_checklist() -> list[dict]:
    if not CHECKLIST_PATH.exists():
        return []
    return json.loads(CHECKLIST_PATH.read_text(encoding="utf-8"))


def validate(item: ChecklistItem, answer: Answer) -> list[str]:
    """Every rule that fails, in plain words. Empty list = allowed to go on
    to approval. Returns all problems rather than the first, so a person
    sees the full picture at once."""
    problems: list[str] = []

    if not answer.answered:
        problems.append("the question wasn't answered, so there is nothing to act on")
    elif answer.confidence is None or answer.confidence.band == "low":
        problems.append("answer confidence is LOW; too weak to act on")

    known = {c.marker for c in answer.citations}
    invented = [m for m in item.citations if m not in known]
    if invented:
        problems.append(f"cites sources the answer doesn't have: {', '.join(invented)}")

    existing = {e["title"].strip().lower() for e in _load_checklist()}
    if item.title.strip().lower() in existing:
        problems.append(f"an item titled '{item.title}' is already on the checklist")

    return problems


# --- Step 4: execute (step 3, approval, happens in the CLI) -------------------


def execute(item: ChecklistItem, answer: Answer) -> dict:
    """Append the approved item to the checklist file. Each entry keeps the
    question, the full source lines and the confidence, so anyone reading
    the checklist later can trace it back to the filing."""
    by_marker = {c.marker: c for c in answer.citations}
    entry = {
        "id": uuid.uuid4().hex[:8],
        "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": answer.query,
        "title": item.title,
        "severity": item.severity,
        "summary": item.summary,
        "sources": [
            {"marker": m, "source": by_marker[m].source, "chunk_id": by_marker[m].chunk_id}
            for m in item.citations
        ],
        "confidence": {"score": answer.confidence.score, "band": answer.confidence.band},
    }
    checklist = _load_checklist()
    checklist.append(entry)
    CHECKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKLIST_PATH.write_text(json.dumps(checklist, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry