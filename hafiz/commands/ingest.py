"""hafiz ingest — index files into the chunks table + optional graph extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from hafiz.core.chunker import chunk_file, walk_and_chunk
from hafiz.core.config import get_settings
from hafiz.core.database import close_engine
from hafiz.core.embeddings import embed_texts
from hafiz.core.store import delete_chunks_for_file, store_chunks

logger = logging.getLogger(__name__)
console = Console()

# Max batch size for embedding calls
EMBED_BATCH_SIZE = 64


def _emit(event: dict) -> None:
    """Write a JSON event to stdout and flush immediately."""
    print(json.dumps(event), flush=True)


async def _do_prune(*, project: str | None = None, output_json: bool = False) -> None:
    """Run prune before ingest — remove chunks for deleted files."""
    from hafiz.commands.prune import _do_prune as prune_impl

    result = await prune_impl(project=project, dry_run=False)
    stale_count = len(result["stale_files"])
    chunks_deleted = result["chunks_deleted"]

    if output_json:
        _emit({
            "event": "prune",
            "status": "done",
            "stale_files": stale_count,
            "chunks_deleted": chunks_deleted,
        })
    elif stale_count:
        console.print(
            f"[yellow]Pruned {chunks_deleted} chunks from {stale_count} stale files.[/yellow]"
        )
    else:
        console.print("[dim]Prune: no stale files found.[/dim]")


def run_ingest(
    path: str,
    *,
    project: str | None = None,
    no_extract: bool = False,
    prune: bool = False,
    output_json: bool = False,
) -> None:
    """Run the ingestion pipeline for a path."""

    async def _ingest():
        try:
            if prune:
                await _do_prune(project=project, output_json=output_json)
            return await _do_ingest(
                path, project=project, no_extract=no_extract, output_json=output_json
            )
        finally:
            await close_engine()

    asyncio.run(_ingest())


async def _do_ingest(
    path: str,
    *,
    project: str | None = None,
    no_extract: bool = False,
    output_json: bool = False,
) -> None:
    """Async ingestion pipeline: chunk -> embed -> store -> extract."""
    target = Path(path).resolve()
    settings = get_settings()
    ignore_patterns = settings.workspace.ignore

    if not target.exists():
        if output_json:
            _emit({"event": "error", "message": f"Path not found: {target}"})
        else:
            console.print(f"[red]Path not found:[/red] {target}")
        raise SystemExit(1)

    # ── Step 1: Chunk files ───────────────────────────────────────────────
    if output_json:
        _emit({"event": "chunking", "status": "start", "path": str(target)})
    else:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        )
        progress_ctx.__enter__()
        progress_ctx.add_task("Chunking files...", total=None)

    chunks = walk_and_chunk(target, ignore_patterns=ignore_patterns)

    if not output_json:
        progress_ctx.__exit__(None, None, None)

    if not chunks:
        if output_json:
            _emit({"event": "complete", "chunks": 0, "files": 0})
        else:
            console.print("[yellow]No content found to index.[/yellow]")
        return

    files_to_reindex = {c.source_file for c in chunks}

    if output_json:
        _emit({
            "event": "chunking",
            "status": "done",
            "chunks": len(chunks),
            "files": len(files_to_reindex),
        })
    else:
        console.print(f"Found [bold]{len(chunks)}[/bold] chunks from [bold]{target}[/bold]")

    # ── Step 2: Delete existing chunks for re-indexed files ───────────────
    for source_file in files_to_reindex:
        await delete_chunks_for_file(source_file)

    # ── Step 3: Embed chunks in batches ───────────────────────────────────
    all_embeddings: list[list[float]] = []

    if output_json:
        _emit({"event": "embedding", "status": "start", "total": len(chunks)})
    else:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        progress_ctx.__enter__()
        rich_task = progress_ctx.add_task("Embedding chunks...", total=len(chunks))

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        texts = [c.content for c in batch]
        embeddings = await embed_texts(texts)
        all_embeddings.extend(embeddings)

        done = min(i + len(batch), len(chunks))
        if output_json:
            _emit({"event": "embedding", "status": "progress", "done": done, "total": len(chunks)})
        else:
            progress_ctx.update(rich_task, advance=len(batch))

    if not output_json:
        progress_ctx.__exit__(None, None, None)

    # ── Step 4: Store in database ─────────────────────────────────────────
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]

    if output_json:
        _emit({"event": "storing", "status": "start"})
    else:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        )
        progress_ctx.__enter__()
        progress_ctx.add_task("Storing in database...", total=None)

    stored = await store_chunks(chunks, all_embeddings, project=project)

    if output_json:
        _emit({
            "event": "storing",
            "status": "done",
            "stored": stored,
            "files": len(files_to_reindex),
        })
    else:
        progress_ctx.__exit__(None, None, None)
        console.print(
            f"[green]Indexed {stored} chunks[/green] from "
            f"[bold]{len(files_to_reindex)}[/bold] files"
        )

    # ── Step 5: Graph extraction (optional) ───────────────────────────────
    if no_extract:
        if output_json:
            _emit({
                "event": "complete",
                "chunks": stored,
                "files": len(files_to_reindex),
                "entities": 0,
                "relations": 0,
            })
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        if output_json:
            _emit({"event": "extraction", "status": "skipped", "reason": "ANTHROPIC_API_KEY not set"})
            _emit({
                "event": "complete",
                "chunks": stored,
                "files": len(files_to_reindex),
                "entities": 0,
                "relations": 0,
            })
        else:
            console.print(
                "[yellow]ANTHROPIC_API_KEY not set — skipping graph extraction.[/yellow]\n"
                "[dim]Set the key or use --no-extract to silence this warning.[/dim]"
            )
        return

    from hafiz.core.extractor import EXTRACTION_BATCH_SIZE, run_extraction

    chunk_dicts = [
        {
            "content": c.content,
            "source_file": c.source_file,
            "language": c.language or "",
            "chunk_id": cid,
        }
        for c, cid in zip(chunks, chunk_ids)
    ]

    if output_json:
        _emit({"event": "extraction", "status": "start", "total": len(chunk_dicts)})
    else:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        progress_ctx.__enter__()
        rich_task = progress_ctx.add_task(
            "Extracting entities & relations...", total=len(chunk_dicts)
        )

    extracted_so_far = 0

    def _on_progress(batch_size: int) -> None:
        nonlocal extracted_so_far
        extracted_so_far += batch_size
        if output_json:
            _emit({
                "event": "extraction",
                "status": "progress",
                "done": extracted_so_far,
                "total": len(chunk_dicts),
            })
        else:
            progress_ctx.update(rich_task, advance=batch_size)

    ent_count, rel_count = await run_extraction(
        chunk_dicts,
        project=project,
        on_progress=_on_progress,
    )

    if not output_json:
        progress_ctx.__exit__(None, None, None)

    if output_json:
        _emit({
            "event": "complete",
            "chunks": stored,
            "files": len(files_to_reindex),
            "entities": ent_count,
            "relations": rel_count,
        })
    else:
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
