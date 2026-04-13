"""hafiz ingest — index files into the chunks table."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from hafiz.core.chunker import walk_and_chunk, chunk_file
from hafiz.core.config import get_settings
from hafiz.core.database import close_engine
from hafiz.core.embeddings import embed_texts
from hafiz.core.store import delete_chunks_for_file, store_chunks

console = Console()

# Max batch size for embedding calls
EMBED_BATCH_SIZE = 64


def run_ingest(path: str, *, project: str | None = None) -> None:
    """Run the ingestion pipeline for a path."""

    async def _ingest():
        try:
            return await _do_ingest(path, project=project)
        finally:
            await close_engine()

    asyncio.run(_ingest())


async def _do_ingest(path: str, *, project: str | None = None) -> None:
    """Async ingestion pipeline: chunk -> embed -> store."""
    target = Path(path).resolve()
    settings = get_settings()
    ignore_patterns = settings.workspace.ignore

    if not target.exists():
        console.print(f"[red]Path not found:[/red] {target}")
        raise SystemExit(1)

    # Step 1: Chunk files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Chunking files...", total=None)
        chunks = walk_and_chunk(
            target,
            ignore_patterns=ignore_patterns,
        )

    if not chunks:
        console.print("[yellow]No content found to index.[/yellow]")
        return

    console.print(f"Found [bold]{len(chunks)}[/bold] chunks from [bold]{target}[/bold]")

    # Step 2: Delete existing chunks for files being re-indexed
    files_to_reindex = {c.source_file for c in chunks}
    for source_file in files_to_reindex:
        await delete_chunks_for_file(source_file)

    # Step 3: Embed chunks in batches
    all_embeddings: list[list[float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(chunks))

        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]
            embeddings = await embed_texts(texts)
            all_embeddings.extend(embeddings)
            progress.update(task, advance=len(batch))

    # Step 4: Store in database
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Storing in database...", total=None)
        stored = await store_chunks(chunks, all_embeddings, project=project)

    console.print(
        f"[green]Indexed {stored} chunks[/green] from "
        f"[bold]{len(files_to_reindex)}[/bold] files"
    )
