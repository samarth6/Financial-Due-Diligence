"""Command line entry points for the pipeline.

    python -m dd.cli chunk      # text -> data/processed/chunks.jsonl
    python -m dd.cli index      # chunks -> Qdrant + BM25
    python -m dd.cli stats      # what is currently indexed
    python -m dd.cli probe "…"  # raw output of each index, unfused
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .config import BM25_PATH, CHUNKS_PATH, filing_text_path, get_config
from .models import Chunk

app = typer.Typer(add_completion=False, help="Financial due-diligence RAG pipeline.")
console = Console()


@app.command()
def chunk() -> None:
    """Layer 2a: parse the filing into citation-carrying chunks."""
    from .chunking import build_chunks, write_chunks

    cfg = get_config()
    path = filing_text_path(cfg)
    if not path.exists():
        raise typer.BadParameter(
            f"no filing text at {path} - run Layer 1 ingestion first"
        )

    text = path.read_text(encoding="utf-8")
    chunks = build_chunks(text, cfg)
    write_chunks(chunks, CHUNKS_PATH)

    lens = sorted(len(c.text) for c in chunks)
    console.print(f"[green]{len(chunks)}[/] chunks -> {CHUNKS_PATH}")
    console.print(
        f"  chars: min {lens[0]} median {lens[len(lens) // 2]} max {lens[-1]} "
        f"(target {cfg.chunking.target_chars}, ceiling {cfg.chunking.max_chars})"
    )
    _item_table(chunks)


@app.command()
def index() -> None:
    """Layer 2b/c: embed chunks into Qdrant and build the BM25 index."""
    from .indexing import build_all

    cfg = get_config()
    console.print(f"embedding with [cyan]{cfg.embedding.model}[/] (first run downloads it)")
    result = build_all(cfg)
    console.print(
        f"[green]indexed {result['vectors']} chunks[/] into "
        f"collection '{cfg.vector_store.collection}' + BM25 at {BM25_PATH}"
    )


@app.command()
def stats() -> None:
    """Show what is on disk and what is indexed."""
    from .chunking import read_chunks
    from .indexing import get_qdrant_client, load_bm25

    cfg = get_config()
    chunks = read_chunks(CHUNKS_PATH)
    console.print(f"chunks on disk: [green]{len(chunks)}[/] ({CHUNKS_PATH})")

    try:
        bm25 = load_bm25()
        console.print(f"BM25 index:     [green]{len(bm25.chunk_ids)}[/] documents")
    except FileNotFoundError as exc:
        console.print(f"BM25 index:     [red]{exc}[/]")

    client = get_qdrant_client(cfg)
    name = cfg.vector_store.collection
    if client.collection_exists(name):
        info = client.get_collection(name)
        console.print(f"Qdrant '{name}': [green]{info.points_count}[/] vectors")
    else:
        console.print(f"Qdrant '{name}': [red]missing - run `index`[/]")
    _item_table(chunks)


@app.command()
def heal(query: str) -> None:
    """Layer 4: self-healing retrieval loop, with full attempt history."""
    from .self_healing import self_heal

    o = self_heal(query)
    console.print(f"[bold]succeeded:[/] {o.succeeded}")
    console.print()
    console.print("[bold]attempts:[/]")
    for a in o.attempts:
        console.print(f"  #{a.attempt} [{a.strategy}] {a.query!r} -> {a.n_relevant}/{a.n_retrieved} relevant")
    console.print()
    if not o.succeeded:
        console.print(f"[red]abstain:[/] {o.abstain_reason}")
        return

    console.print("[bold]graded chunks:[/]")
    for i, g in enumerate(o.chunks):
        c = g.retrieved.chunk
        console.print(f"  [{i}] score={g.relevance:.2f}  {c.item}/{c.subsection[:30]}")
        console.print(f"      reason: {g.reason}")


@app.command()
def probe(query: str, k: int = 5) -> None:
    """Show each index's raw top-k for a query, side by side.

    Deliberately unfused. Layer 3 owns RRF and re-ranking; this exists to
    confirm Layer 2 built something searchable, and to make the two indexes'
    different failure modes visible before they are blended together.
    """
    from .chunking import read_chunks
    from .embedding import embed_query
    from .indexing import get_qdrant_client, load_bm25

    cfg = get_config()
    by_id = {c.chunk_id: c for c in read_chunks(CHUNKS_PATH)}

    client = get_qdrant_client(cfg)
    hits = client.query_points(
        collection_name=cfg.vector_store.collection,
        query=embed_query(query, cfg).tolist(),
        limit=k,
    ).points
    dense = [(h.payload["chunk_id"], h.score) for h in hits]
    sparse = load_bm25().search(query, k)

    for title, rows in (("dense (vector)", dense), ("sparse (BM25)", sparse)):
        table = Table(title=title, show_lines=False)
        table.add_column("#", width=3)
        table.add_column("score", width=8)
        table.add_column("where", width=34)
        table.add_column("text")
        for rank, (cid, score) in enumerate(rows, 1):
            c = by_id.get(cid)
            where = f"{c.item} / {c.subsection[:22]}" if c else cid
            snippet = " ".join(c.text.split())[:90] if c else ""
            table.add_row(str(rank), f"{score:.4f}", where, snippet)
        console.print(table)

    overlap = {c for c, _ in dense} & {c for c, _ in sparse}
    console.print(
        f"both indexes agreed on [cyan]{len(overlap)}[/]/{k} chunks "
        "(this overlap is the hybrid_agreement signal Layer 5 scores on)"
    )

@app.command()
def retrieve(query: str) -> None:
    """Layer 3: fused, reranked retrieval — the real answer, not probe's raw comparison."""
    from .retrieval import hybrid_retrieve

    results = hybrid_retrieve(query)

    table = Table(title=f"hybrid_retrieve: {query!r}", show_lines=False)
    table.add_column("#", width=3)
    table.add_column("rerank", width=8)
    table.add_column("rrf", width=8)
    table.add_column("v_rank", width=6)
    table.add_column("b_rank", width=6)
    table.add_column("where", width=34)
    table.add_column("text")

    for rank, r in enumerate(results, 1):
        c = r.chunk
        where = f"{c.item} / {c.subsection[:22]}"
        snippet = " ".join(c.text.split())[:90]
        v_rank = str(r.vector_rank) if r.vector_rank else "-"
        b_rank = str(r.bm25_rank) if r.bm25_rank else "-"
        rerank_str = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "-"
        table.add_row(str(rank), rerank_str, f"{r.rrf_score:.4f}", v_rank, b_rank, where, snippet)

    console.print(table)


@app.command()
def answer(query: str) -> None:
    """Layer 5: full pipeline - cited answer with confidence."""
    from .generation import answer as run_answer

    a = run_answer(query)
    if not a.answered:
        console.print(f"[red]no answer:[/] {a.text}")
        console.print(f"[dim]({a.elapsed_s}s)[/]")
        return
    console.print(a.text)
    console.print()
    console.print("[bold]sources:[/]")
    for c in a.citations:
        console.print(f"  {c.marker} {c.source}")
    cf = a.confidence
    console.print()
    console.print(
        f"[bold]confidence:[/] {cf.band.upper()} ({cf.score})  "
        f"hybrid={cf.hybrid_agreement} rerank={cf.rerank_strength} loop={cf.loop_cleanliness}"
    )
    for n in cf.notes:
        console.print(f"  note: {n}")
    console.print(f"[dim]({a.elapsed_s}s)[/]")


def _item_table(chunks: list[Chunk]) -> None:
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.item] = counts.get(c.item, 0) + 1
    table = Table("item", "section", "chunks", title="coverage by 10-K item")
    sections = {c.item: c.section for c in chunks}
    for item in sorted(counts, key=lambda i: (len(i), i)):
        table.add_row(item or "(none)", sections.get(item, "")[:46], str(counts[item]))
    console.print(table)


if __name__ == "__main__":
    app()