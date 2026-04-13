"""hafiz ingest — index files into the chunks table + optional graph extraction."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from hafiz.core.chunker import walk_and_chunk, chunk_file
from hafiz.core.config import get_settings
from hafiz.core.database import close_engine
from hafiz.core.embeddings import embed_texts
from hafiz.core.store import delete_chunks_for_file, store_chunks

logger = logging.getLogger(__name__)
console = Console()

# Max batch size for embedding calls
EMBED_BATCH_SIZE = 64


def run_ingest(
    path: str,
    *,
    project: str | None = None,
    no_extract: bool = False,
) -> None:
    """Run the ingestion pipeline for a path."""

    async def _ingest():
        try:
            return await _do_ingest(path, project=project, no_extract=no_extract)
        finally:
            await close_engine()

    asyncio.run(_ingest())


async def _do_ingest(
    path: str,
    *,
    project: str | None = None,
    no_extract: bool = False,
) -> None:
    """Async ingestion pipeline: chunk -> embed -> store -> extract."""
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

    # Step 4: Store in database — generate chunk IDs so we can reference them in extraction
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]

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

    # Step 5: Graph extraction (optional)
    if no_extract:
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[yellow]ANTHROPIC_API_KEY not set — skipping graph extraction.[/yellow]\n"
            "[dim]Set the key or use --no-extract to silence this warning.[/dim]"
        )
        return

    from hafiz.core.extractor import run_extraction, EXTRACTION_BATCH_SIZE

    # Build chunk dicts for the extractor
    chunk_dicts = [
        {
            "content": c.content,
            "source_file": c.source_file,
            "language": c.language or "",
            "chunk_id": cid,
        }
        for c, cid in zip(chunks, chunk_ids)
    ]

    total_batches = (len(chunk_dicts) + EXTRACTION_BATCH_SIZE - 1) // EXTRACTION_BATCH_SIZE

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting entities & relations...", total=len(chunk_dicts))

        def _on_progress(batch_size: int) -> None:
            progress.update(task, advance=batch_size)

        ent_count, rel_count = await run_extraction(
            chunk_dicts,
            project=project,
            on_progress=_on_progress,
        )

    if ent_count or rel_count:
        console.print(
            f"[green]Extracted {ent_count} entities, {rel_count} relations[/green]"
        )
    else:
        console.print("[dim]No entities or relations extracted.[/dim]")


def run_git_hook_ingest_cmd(*, project: str | None = None) -> None:
    """Run git-hook-based ingest: only files changed in the latest commit."""
    from hafiz.core.git_hooks import run_git_hook_ingest

    async def _ingest():
        try:
            return await run_git_hook_ingest(".", project=project)
        finally:
            await close_engine()

    files_processed, chunks_stored = asyncio.run(_ingest())

    console.print(
        f"[green]Git hook ingest:[/green] {files_processed} files processed, "
        f"{chunks_stored} chunks stored."
    )
