"""hafiz extract — import and run entity/relation extraction.

``extract import``: agent-driven — Claude Code (or any LLM) analyses chunks
exported by ``hafiz chunks export`` and produces a JSON payload.

``extract run``: API-driven — calls the Anthropic API to extract entities from
chunks that have no corresponding entities yet (incremental).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from hafiz.core.database import close_engine
from hafiz.core.extractor import (
    EXTRACTION_BATCH_SIZE,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
    store_extraction,
)

console = Console()


def _parse_extraction_json(data: dict) -> ExtractionResult:
    """Parse a JSON payload into an ExtractionResult."""
    entities = [
        ExtractedEntity(
            name=e["name"],
            entity_type=e["entity_type"],
            description=e.get("description", ""),
            source_file=e.get("source_file"),
            chunk_id=e.get("chunk_id"),
        )
        for e in data.get("entities", [])
    ]
    relations = [
        ExtractedRelation(
            source_name=r["source_name"],
            source_type=r["source_type"],
            target_name=r["target_name"],
            target_type=r["target_type"],
            relation_type=r["relation_type"],
            evidence=r.get("evidence", ""),
        )
        for r in data.get("relations", [])
    ]
    return ExtractionResult(entities=entities, relations=relations)


def run_extract_import(
    file: str | None = None,
    *,
    project: str | None = None,
) -> None:
    """Import extraction results from a JSON file or stdin."""

    async def _run():
        try:
            if file:
                with open(file) as f:
                    data = json.load(f)
            else:
                data = json.load(sys.stdin)

            result = _parse_extraction_json(data)
            ent_count, rel_count = await store_extraction(result, project=project)
            console.print(
                f"[green]Imported {ent_count} entities, {rel_count} relations[/green]"
            )
        finally:
            await close_engine()

    asyncio.run(_run())


async def _find_unextracted_chunks(
    project: str | None = None,
) -> list[dict]:
    """Find chunks whose source files have no corresponding entities.

    Returns chunk dicts suitable for run_extraction().
    """
    from sqlalchemy import select, func
    from hafiz.core.database import Chunk, Entity, get_session_factory

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Source files that have entities
        entity_files_stmt = (
            select(Entity.source_file)
            .where(Entity.source_file.isnot(None))
        )
        if project:
            entity_files_stmt = entity_files_stmt.where(Entity.project == project)
        entity_files_result = await session.execute(entity_files_stmt)
        files_with_entities = {row[0] for row in entity_files_result.all()}

        # All chunks, filtered by project
        chunk_stmt = select(Chunk).where(Chunk.source_file.isnot(None))
        if project:
            chunk_stmt = chunk_stmt.where(Chunk.project == project)
        chunk_result = await session.execute(chunk_stmt)
        all_chunks = chunk_result.scalars().all()

        # Filter to chunks whose source files have no entities
        unextracted = [
            {
                "content": c.content,
                "source_file": c.source_file,
                "language": c.language or "",
                "chunk_id": str(c.id),
            }
            for c in all_chunks
            if c.source_file not in files_with_entities
        ]

        return unextracted


def run_extract_run(
    *,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Run LLM extraction on chunks that don't have entities yet."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]ANTHROPIC_API_KEY not set.[/red] "
            "Required for API-driven extraction."
        )
        raise SystemExit(1)

    async def _run():
        try:
            from hafiz.core.extractor import run_extraction

            chunks = await _find_unextracted_chunks(project=project)

            if not chunks:
                if output_json:
                    print(json.dumps({
                        "event": "extract_run",
                        "status": "done",
                        "chunks": 0,
                        "entities": 0,
                        "relations": 0,
                        "message": "All chunks already have entities.",
                    }), flush=True)
                else:
                    console.print(
                        "[green]All chunks already have entities. Nothing to extract.[/green]"
                    )
                return

            if output_json:
                print(json.dumps({
                    "event": "extract_run",
                    "status": "start",
                    "chunks": len(chunks),
                }), flush=True)
            else:
                proj_label = f" for project '{project}'" if project else ""
                console.print(
                    f"Found [bold]{len(chunks)}[/bold] chunks without entities{proj_label}."
                )

            extracted_so_far = 0

            if output_json:
                def _on_progress(batch_size: int):
                    nonlocal extracted_so_far
                    extracted_so_far += batch_size
                    print(json.dumps({
                        "event": "extract_run",
                        "status": "progress",
                        "done": extracted_so_far,
                        "total": len(chunks),
                    }), flush=True)
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
                    "Extracting entities & relations...", total=len(chunks)
                )

                def _on_progress(batch_size: int):
                    nonlocal extracted_so_far
                    extracted_so_far += batch_size
                    progress_ctx.update(rich_task, advance=batch_size)

            ent_count, rel_count = await run_extraction(
                chunks, project=project, on_progress=_on_progress,
            )

            if not output_json:
                progress_ctx.__exit__(None, None, None)

            if output_json:
                print(json.dumps({
                    "event": "extract_run",
                    "status": "done",
                    "chunks": len(chunks),
                    "entities": ent_count,
                    "relations": rel_count,
                }), flush=True)
            else:
                console.print(
                    f"[green]Extracted {ent_count} entities, "
                    f"{rel_count} relations[/green] "
                    f"from {len(chunks)} chunks."
                )
        finally:
            await close_engine()

    asyncio.run(_run())
