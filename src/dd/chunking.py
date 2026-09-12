
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ChunkingCfg, Config, get_config
from .models import Chunk, SourceKind

# "Item 1A. Risk Factors" at column 0. The TOC's copies are indented, so
# anchoring to the line start is what separates the real heading from its
# table-of-contents echo.
ITEM_RE = re.compile(r"^Item\s+(\d{1,2}[A-C]?)\.\s*(.*)$")
PART_RE = re.compile(r"^PART\s+([IVX]+)\s*$")

# "Apple Inc. | 2025 Form 10-K | 5" - a page footer, not content.
FOOTER_RE = re.compile(r"^.{0,40}\|\s*\d{4}\s+Form\s+10-[KQ]\s*\|\s*\d+\s*$")

# Box-drawing rules and 3+ space columns are how the text dump renders tables.
# The range is the Unicode box-drawing block (U+2500-U+257F), built with chr()
# so the pattern cannot be mangled by an encoding round-trip of this file.
_BOX_RANGE = chr(0x2500) + "-" + chr(0x257F)
TABLE_RULE_RE = re.compile("[" + _BOX_RANGE + "]{3,}|_{6,}")
COLUMN_GAP_RE = re.compile(r"\S {3,}\S")

# Sentence boundary for splitting paragraphs that exceed the target size on
# their own. Avoids breaking on "U.S." or "Inc." by requiring a following
# capital or quote and a preceding lowercase/digit/closing punctuation.
SENTENCE_RE = re.compile(r'(?<=[a-z0-9\)\]"”])\.\s+(?=[A-Z"“])')

MAX_HEADING_CHARS = 130


@dataclass
class Block:
    """One structural unit of the filing: a paragraph, a heading, or a table."""

    text: str
    kind: str  # "prose" | "table"
    part: str
    item: str
    section: str
    subsection: str
    char_start: int
    char_end: int


def _is_table_line(line: str) -> bool:
    """A table line either draws a rule or has multiple wide column gaps."""
    if TABLE_RULE_RE.search(line):
        return True
    gaps = len(COLUMN_GAP_RE.findall(line))
    if gaps >= 2:
        return True
    # A single wide gap still reads as tabular when the tail is numeric,
    # e.g. "  Total operating expenses      $  62,151    8%".
    return gaps == 1 and bool(re.search(r"[\d$%]", line.split("   ")[-1]))


def _is_heading(line: str) -> bool:
    """Short, title-ish, no terminal period - a subsection heading.

    Risk Factors also uses full-sentence headings ("The Company's business can
    be impacted by political events...") which end in a period and are treated
    as prose. That is deliberate: they read as the opening sentence of their own
    chunk, which is exactly where a retriever wants them.
    """
    s = line.strip()
    if not s or len(s) > MAX_HEADING_CHARS:
        return False
    if s.endswith((".", ";", ":", ",")):
        return False
    if _is_table_line(line):
        return False
    # Headings are not indented in this dump; table fragments usually are.
    if line[:1].isspace():
        return False
    words = s.split()
    if len(words) > 18:
        return False
    # Require at least one capitalised word and no sentence-like verb pileup.
    return any(w[:1].isupper() for w in words)


def find_body_start(lines: list[str]) -> int:
    """Index of the first real "PART I" line, skipping the cover page and TOC.

    The TOC lists "Part I" indented; the body's is at column 0. Falling back to
    0 would silently index the whole cover page, so the fallback is loud.
    """
    for i, line in enumerate(lines):
        if PART_RE.match(line):
            return i
    raise ValueError(
        "no 'PART I' line at column 0 - the text dump's shape changed, "
        "re-check the Layer 1 extraction before indexing"
    )


def parse_blocks(text: str) -> list[Block]:
    """Walk the filing top to bottom, emitting structure-tagged blocks."""
    lines = text.split("\n")
    start = find_body_start(lines)

    # Byte offsets into the original text, so a citation can point back at the
    # source span even after footers and blank lines are stripped.
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    blocks: list[Block] = []
    part = item = section = subsection = ""
    buf: list[str] = []
    buf_kind = "prose"
    buf_start = 0

    def flush() -> None:
        nonlocal buf, buf_kind, buf_start
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            blocks.append(
                Block(
                    text=body,
                    kind=buf_kind,
                    part=part,
                    item=item,
                    section=section,
                    subsection=subsection,
                    char_start=buf_start,
                    char_end=buf_start + len("\n".join(buf)),
                )
            )
        buf = []
        buf_kind = "prose"

    for i in range(start, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if not stripped or FOOTER_RE.match(stripped):
            flush()
            continue

        if m := PART_RE.match(line):
            flush()
            part = f"Part {m.group(1)}"
            item = section = subsection = ""
            continue

        if m := ITEM_RE.match(line):
            flush()
            item = f"Item {m.group(1)}"
            section = m.group(2).strip() or item
            subsection = ""
            continue

        kind = "table" if _is_table_line(line) else "prose"

        if kind == "prose" and _is_heading(line):
            flush()
            subsection = stripped
            continue

        # A table run and a prose run are never merged into one block.
        if buf and kind != buf_kind:
            flush()
        if not buf:
            buf_start = offsets[i]
            buf_kind = kind
        buf.append(line if kind == "table" else stripped)

    flush()
    return blocks


def _split_long(text: str, target: int, overlap: int) -> list[str]:
    """Split one oversized paragraph on sentence boundaries, with overlap.

    Overlap exists so a claim that straddles a boundary survives in at least
    one chunk intact.
    """
    sentences = SENTENCE_RE.split(text)
    if len(sentences) == 1:
        # No sentence boundary to use (a long single sentence or a URL blob):
        # fall back to hard character slicing rather than emitting a chunk the
        # embedding model will silently truncate.
        step = max(target - overlap, 1)
        return [text[i : i + target] for i in range(0, len(text), step)]

    out: list[str] = []
    cur: list[str] = []
    size = 0
    for sent in sentences:
        s = sent if sent.endswith(".") else sent + "."
        if size + len(s) > target and cur:
            out.append(" ".join(cur))
            # Carry the tail of the previous chunk forward as overlap.
            carry: list[str] = []
            carried = 0
            for prev in reversed(cur):
                if carried + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev)
            cur = carry
            size = carried
        cur.append(s)
        size += len(s)
    if cur:
        out.append(" ".join(cur))
    return out


def _split_table(text: str, max_chars: int, title: str = "") -> list[str]:
    """Row-split an oversized table, repeating its header on every piece.

    A table has to be split on line boundaries or numbers get separated from
    their row labels. Repeating the header means each piece still says which
    fiscal years its columns belong to - without that, a retrieved fragment
    reads "34,550  10%" with no way to know what either number is.
    """
    lines = text.split("\n")
    # Header = leading lines up to and including the column rule. Only lines
    # above that rule are a real header; if the block has no rule (it is
    # already a continuation of a table split elsewhere), fabricating one from
    # the first data row would repeat a number as if it were a label, so fall
    # back to the table's title instead.
    header_end = 0
    for i, line in enumerate(lines[:4]):
        if TABLE_RULE_RE.search(line):
            header_end = i + 1
            break
    header = lines[:header_end] if header_end else ([title] if title else [])
    header_len = sum(len(h) + 1 for h in header)
    budget = max(max_chars - header_len, max_chars // 2)

    pieces: list[str] = []
    cur: list[str] = []
    size = 0
    for line in lines[header_end:]:
        if size + len(line) + 1 > budget and cur:
            pieces.append("\n".join(header + cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        pieces.append("\n".join(header + cur))
    return pieces or [text]


def chunk_blocks(blocks: list[Block], cfg: ChunkingCfg) -> list[tuple[str, Block]]:
    """Pack blocks into ~target_chars chunks without crossing a heading.

    Chunks never span two subsections. That costs a little packing efficiency
    and buys a citation that is actually true.
    """
    out: list[tuple[str, Block]] = []
    cur: list[Block] = []
    size = 0

    def flush_group() -> None:
        nonlocal cur, size
        if not cur:
            return
        head = cur[0]
        merged = "\n\n".join(b.text for b in cur)
        out.append((merged, head))
        cur = []
        size = 0

    for block in blocks:
        if block.item in cfg.exclude_items:
            flush_group()
            continue

        # Tables are never packed with prose. They are only split when they
        # exceed the embedding model's real input limit, and then by row.
        if block.kind == "table":
            flush_group()
            if len(block.text) > cfg.max_chars:
                title = block.subsection or block.section
                for piece in _split_table(block.text, cfg.max_chars, title):
                    out.append((piece, block))
            else:
                out.append((block.text, block))
            continue

        same_section = bool(cur) and (
            cur[0].item == block.item and cur[0].subsection == block.subsection
        )
        if cur and not same_section:
            flush_group()

        if len(block.text) > cfg.target_chars:
            flush_group()
            for piece in _split_long(block.text, cfg.target_chars, cfg.overlap_chars):
                out.append((piece, block))
            continue

        if size + len(block.text) > cfg.target_chars and cur:
            flush_group()
        cur.append(block)
        size += len(block.text)

    flush_group()
    return out


def build_chunks(text: str, cfg: Config | None = None) -> list[Chunk]:
    """Text in, citation-carrying Chunks out. The whole of Layer 2a."""
    conf = cfg or get_config()
    blocks = parse_blocks(text)
    packed = chunk_blocks(blocks, conf.chunking)

    fiscal_year = _infer_fiscal_year(text, conf)
    chunks: list[Chunk] = []
    for idx, (body, head) in enumerate(packed):
        if len(body.strip()) < conf.chunking.min_chars:
            continue  # stray heading fragments and one-line table remnants
        chunks.append(
            Chunk(
                chunk_id=f"{conf.company.ticker}-{conf.company.form}-{fiscal_year}-{idx:04d}",
                text=body,
                kind=SourceKind.NARRATIVE,
                company=conf.company.name,
                ticker=conf.company.ticker,
                form=conf.company.form,
                fiscal_year=fiscal_year,
                item=head.item,
                section=head.section,
                subsection=head.subsection,
                char_start=head.char_start,
                char_end=head.char_end,
            )
        )
    return chunks


def _infer_fiscal_year(text: str, cfg: Config) -> str:
    """Read the fiscal year off the cover page, falling back to config."""
    if str(cfg.company.year) != "latest":
        return f"FY{cfg.company.year}"
    m = re.search(r"fiscal year ended\s+\w+\s+\d{1,2},\s*(\d{4})", text[:4000], re.I)
    return f"FY{m.group(1)}" if m else "FY"


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"no chunks at {path} - run `dd chunk` first")
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate(json.loads(line)) for line in fh if line.strip()]
