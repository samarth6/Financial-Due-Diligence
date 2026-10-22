"""Tests for Layer 2a, the chunker (src/dd/chunking.py).

The chunker runs on judgement rules tuned by eye to Apple's 10-K: what counts
as a heading, a page footer, a table-of-contents line, a table. A small edit
to any rule can silently re-cut and re-label the chunks - with no error, and
with chunk IDs (which are position numbers) suddenly pointing at different
text. These tests make that kind of change fail loudly instead.

Two groups:
1. The real filing: the output we know is correct, pinned down.
2. Each rule on its own, on small hand-written examples.

If a test in group 1 fails after a change you MEANT to make, check the new
output by hand (`dd chunk`), then update the expected value here - and re-run
`dd index`, because the search indexes still hold the old chunk IDs.

Run from the project root:   pytest
"""

from __future__ import annotations

import re

import pytest

import dd.chunking as ch
from dd.config import ChunkingCfg, filing_text_path, get_config

# --- group 1: the real filing ---------------------------------------------

EXPECTED_CHUNK_COUNT = 385

EXPECTED_ITEMS = {
    "Item 1", "Item 1A", "Item 1C", "Item 2", "Item 3", "Item 5", "Item 7",
    "Item 7A", "Item 8", "Item 9A", "Item 9B", "Item 10", "Item 11",
    "Item 12", "Item 13", "Item 14", "Item 16",
}

# "" is the introduction to Risk Factors, before its first subheading.
EXPECTED_RISK_FACTOR_SUBSECTIONS = {
    "",
    "Macroeconomic and Industry Risks",
    "Business Risks",
    "Legal and Regulatory Compliance Risks",
    "Financial Risks",
    "General Risks",
}

# Anything that looks like a page footer: "Apple Inc. | 2025 Form 10-K | 5".
FOOTER_LIKE = re.compile(r"Form\s+10-?K\s*\|\s*\d+")
# A table-of-contents line: indented Item, title, then a page number.
TOC_LINE = re.compile(r"^\s+Item\s+\d{1,2}[A-C]?\.\s+.+\s{2,}\d+\s*$", re.MULTILINE)


@pytest.fixture(scope="module")
def chunks():
    cfg = get_config()
    return ch.build_chunks(filing_text_path(cfg).read_text(encoding="utf-8"), cfg)


def test_chunk_count_unchanged(chunks):
    assert len(chunks) == EXPECTED_CHUNK_COUNT, (
        f"the filing now produces {len(chunks)} chunks, not {EXPECTED_CHUNK_COUNT}. "
        "Chunk IDs are position numbers, so every ID after the first changed "
        "chunk now points at different text. If this change is intended, "
        "check the output, update EXPECTED_CHUNK_COUNT and re-run `dd index`."
    )


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_every_chunk_belongs_to_a_known_item(chunks):
    untagged = [c.chunk_id for c in chunks if not c.item]
    assert not untagged, f"chunks with no Item (cover page or TOC leaked in?): {untagged[:5]}"
    assert {c.item for c in chunks} == EXPECTED_ITEMS


def test_no_page_footer_anywhere(chunks):
    """If the footer rule stops matching, footers don't show up as junk text -
    they pass the heading rule and become subsection NAMES, so citations read
    "... / Apple Inc. | 2025 Form 10-K | 2". Both places are checked."""
    in_text = [c.chunk_id for c in chunks if FOOTER_LIKE.search(c.text)]
    in_label = sorted({c.subsection for c in chunks if FOOTER_LIKE.search(c.subsection)})
    assert not in_text, f"page footers inside chunk text: {in_text[:5]}"
    assert not in_label, f"page footers used as subsection names: {in_label[:5]}"


def test_subsection_names_look_like_headings(chunks):
    bad = sorted({
        c.subsection for c in chunks
        if c.subsection and (
            len(c.subsection) > ch.MAX_HEADING_CHARS
            or c.subsection.endswith(".")
            or "|" in c.subsection
        )
    })
    assert not bad, f"subsection names that are not headings: {bad[:5]}"


def test_no_table_of_contents_or_cover_page(chunks):
    toc = [c.chunk_id for c in chunks if TOC_LINE.search(c.text)]
    cover = [c.chunk_id for c in chunks if "Commission File Number" in c.text]
    assert not toc, f"table-of-contents lines in chunks: {toc[:5]}"
    assert not cover, f"cover-page text in chunks: {cover[:5]}"


def test_risk_factor_subsections(chunks):
    found = {c.subsection for c in chunks if c.item == "Item 1A"}
    assert found == EXPECTED_RISK_FACTOR_SUBSECTIONS


def test_known_passage_is_filed_in_the_right_place(chunks):
    """Spot-check one passage end to end: found, and labelled correctly."""
    passage = "The Company has a large, global business with sales outside the U.S."
    hits = [c for c in chunks if passage in c.text]
    assert hits, "the trade/tariff risk paragraph is missing from every chunk"
    assert {(c.item, c.subsection) for c in hits} == {("Item 1A", "Macroeconomic and Industry Risks")}
    assert hits[0].citation() == "AAPL | 10-K | FY2025 | Item 1A | Risk Factors"


def test_chunk_sizes_within_limits(chunks):
    cfg = get_config().chunking
    too_big = [(c.chunk_id, len(c.text)) for c in chunks if len(c.text) > cfg.max_chars]
    too_small = [(c.chunk_id, len(c.text)) for c in chunks if len(c.text) < cfg.min_chars]
    assert not too_big, f"over max_chars={cfg.max_chars} (the embedder would truncate): {too_big[:5]}"
    assert not too_small, f"under min_chars={cfg.min_chars}: {too_small[:5]}"


def test_excluded_items_are_absent(chunks):
    excluded = set(get_config().chunking.exclude_items)
    assert not {c.item for c in chunks} & excluded


def test_filing_metadata(chunks):
    assert {(c.company, c.ticker, c.form, c.fiscal_year) for c in chunks} == {
        ("Apple Inc.", "AAPL", "10-K", "FY2025")
    }


# --- group 2: each rule on small hand-written examples ---------------------

FOOTER = "Apple Inc. | 2025 Form 10-K | 5"


def _doc(*body_lines: str) -> str:
    """A minimal filing: cover page, indented TOC, then the real body."""
    return "\n".join([
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
        "",
        "  Part I",
        "  Item 1A.      Risk Factors                                   5",
        "",
        "PART I",
        "",
        "Item 1A. Risk Factors",
        "",
        *body_lines,
    ])


def test_footer_regex_matches_apple_footer():
    assert ch.FOOTER_RE.match(FOOTER)


def test_footer_is_dropped_not_kept_or_made_a_heading():
    blocks = ch.parse_blocks(_doc(
        "Macroeconomic Risks",
        "",
        "First part of a paragraph that is interrupted by a page break.",
        "",
        FOOTER,
        "",
        "Second part of the same discussion after the page break.",
    ))
    assert all(FOOTER not in b.text for b in blocks)
    assert all(b.subsection == "Macroeconomic Risks" for b in blocks), (
        "the footer was treated as a heading and renamed the subsection"
    )


def test_table_of_contents_is_skipped():
    blocks = ch.parse_blocks(_doc("A real paragraph of risk factor text appears here."))
    assert len(blocks) == 1
    assert "UNITED STATES" not in blocks[0].text  # cover page
    assert (blocks[0].item, blocks[0].section) == ("Item 1A", "Risk Factors")


def test_missing_part_line_fails_loudly():
    with pytest.raises(ValueError, match="PART I"):
        ch.find_body_start(["cover page", "  Part I", "  Item 1.   Business   1"])


@pytest.mark.parametrize("line", [
    "Macroeconomic and Industry Risks",
    "Company Background",
    "Legal and Regulatory Compliance Risks",
])
def test_is_heading_true(line):
    assert ch._is_heading(line)


@pytest.mark.parametrize("line, why", [
    ("The Company may be adversely affected by tariffs.", "ends with a full stop - a sentence"),
    ("Current assets:", "ends with a colon - a table row label"),
    ("  Announced Plans or Programs", "indented - a table fragment"),
    ("  Research and development      $   34,550     10%", "a table row"),
    ("x" * 131, "longer than MAX_HEADING_CHARS"),
    ("", "empty"),
])
def test_is_heading_false(line, why):
    assert not ch._is_heading(line), why


def test_known_quirk_lowercase_first_letter_is_not_a_heading():
    """Documented behaviour, not a goal: "iPhone" starts lowercase, so it is
    not taken as a heading and its paragraphs keep the parent subsection
    ("Products"). If you change this on purpose, update this test and
    EXPECTED_CHUNK_COUNT together."""
    assert not ch._is_heading("iPhone")


@pytest.mark.parametrize("line, expected", [
    (" " + chr(0x2500) * 20, True),                                 # box-drawing rule
    ("  Total operating expenses      $  62,151    8%", True),       # one gap, numeric tail
    ("  Net sales      416,161      391,035      383,285", True),    # several column gaps
    ("A normal sentence of prose with ordinary single spaces.", False),
])
def test_is_table_line(line, expected):
    assert ch._is_table_line(line) is expected


def test_tables_are_not_merged_with_prose():
    blocks = ch.parse_blocks(_doc(
        "Operating Expenses",
        "",
        "Operating expenses for 2025 were as follows:",
        "  Research and development      $   34,550     10%",
        "  Selling, general and administrative      $   27,601     6%",
        "The growth was driven by headcount-related expenses.",
    ))
    assert [b.kind for b in blocks] == ["prose", "table", "prose"]


def _block(text: str, item: str = "Item 1A", subsection: str = "Risks", kind: str = "prose") -> ch.Block:
    return ch.Block(text=text, kind=kind, part="Part I", item=item, section="Risk Factors",
                    subsection=subsection, char_start=0, char_end=len(text))


def test_oversized_table_is_split_by_row_with_header_repeated():
    header = ["                       2025       2024", " " + chr(0x2500) * 40]
    rows = [f"  Line item number {i:03d}      ${i * 1000:>8,}   ${i * 900:>8,}" for i in range(120)]
    pieces = ch._split_table("\n".join(header + rows), max_chars=1000)

    assert len(pieces) > 1
    for p in pieces:
        assert len(p) <= 1000
        assert p.startswith("\n".join(header)), "a piece lost the column header"
    body_rows = [line for p in pieces for line in p.split("\n")[len(header):]]
    assert body_rows == rows, "rows were lost, duplicated or reordered"


def test_long_paragraph_is_split_on_sentences_with_overlap():
    sentences = [f"Sentence {i} describes one specific risk in about sixty chars." for i in range(40)]
    pieces = ch._split_long(" ".join(sentences), target=900, overlap=150)

    assert len(pieces) > 1
    assert all(len(p) <= 900 for p in pieces)
    for prev, nxt in zip(pieces, pieces[1:]):
        last_sentence = prev.rsplit("Sentence ", 1)[1]
        assert last_sentence in nxt, "consecutive pieces do not overlap"
    for s in sentences:
        assert any(s in p for p in pieces), f"sentence lost in splitting: {s!r}"


def test_chunks_never_span_two_subsections():
    cfg = ChunkingCfg()
    blocks = [_block("Short text about one topic.", subsection="Business Risks"),
              _block("Short text about another topic.", subsection="Financial Risks")]
    packed = ch.chunk_blocks(blocks, cfg)
    assert len(packed) == 2, "two small blocks from different subsections were merged"


def test_excluded_item_is_dropped():
    cfg = ChunkingCfg(exclude_items=["Item 15"])
    packed = ch.chunk_blocks([_block("Exhibit index text.", item="Item 15")], cfg)
    assert packed == []
