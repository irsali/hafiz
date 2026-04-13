"""hafiz prune — remove stale chunks whose source files no longer exist."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hafiz.core.config import get_settings
from hafiz.core.database import Entity, close_engine, get_session_factory
from hafiz.core.store import delete_chunks_for_file, get_indexed_files

logger = logging.getLogger(__name__)
console = Console()


async def _find_stale_files(project: str | None = None) -> list[str]:
    """Find source files in the index that no longer exist on disk."""
    indexed = await get_indexed_files()
    settings = get_settings()
    workspace_root = Path(settings.workspace.root).resolve()

    stale: list[str] = []
    for source_file in indexed:
        full_path = workspace_root / source_file
        if not full_path.exists():
            stale.append(source_file)

    return stale


async def _mark_entities_stale(source_files: list[str]) -> int:
    """Mark entities as stale if their source_file was pruned."""
    if not source_files:
        return 0

    session_factory = get_session_factory()
    marked = 0

    async with session_factory() as session:
        async with session.begin():
            from sqlalchemy import select, update
            from sqlalchemy.dialects.postgresql import JSONB

            for source_file in source_files:
                stmt = (
                    select(Entity)
                    .where(Entity.source_file == source_file)
                )
                result = await session.execute(stmt)
                entities = result.scalars().all()

                for entity in entities:
                    props = dict(entity.properties) if entity.properties else {}
                    props["stale"] = True
                    entity.properties = props
                    marked += 1

    return marked


async def _do_prune(
    project: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the prune operation. Returns a summary dict."""
    stale_files = await _find_stale_files(project=project)

    if not stale_files:
        return {"stale_files": [], "chunks_deleted": 0, "entities_marked_stale": 0}

    if dry_run:
        return {
            "stale_files": stale_files,
            "chunks_deleted": 0,
            "entities_marked_stale": 0,
            "dry_run": True,
        }

    # Delete stale chunks
    total_deleted = 0
    for source_file in stale_files:
        deleted = await delete_chunks_for_file(source_file)
        total_deleted += deleted
        logger.info("Pruned %d chunks from %s", deleted, source_file)

    # Mark entities as stale
    entities_marked = await _mark_entities_stale(stale_files)

    return {
        "stale_files": stale_files,
        "chunks_deleted": total_deleted,
        "entities_marked_stale": entities_marked,
    }


def run_prune(
    project: str | None = None,
    dry_run: bool = False,
    output_json: bool = False,
) -> None:
    """Run the prune command."""

    async def _prune():
        try:
            return await _do_prune(project=project, dry_run=dry_run)
        finally:
            await close_engine()

    result = asyncio.run(_prune())

    if output_json:
        console.print(json.dumps(result, indent=2))
        return

    stale_files = result["stale_files"]

    if not stale_files:
        console.print("[green]No stale files found. Index is clean.[/green]")
        return

    if dry_run:
        table = Table(title="Stale files (dry run)")
        table.add_column("Source File", style="yellow")
        for f in stale_files:
            table.add_row(f)
        console.print(table)
        console.print(f"\n[yellow]{len(stale_files)} stale files would be pruned.[/yellow]")
    else:
        console.print(
            f"[green]Pruned {result['chunks_deleted']} chunks[/green] "
            f"from [bold]{len(stale_files)}[/bold] stale files."
        )
        if result["entities_marked_stale"]:
            console.print(
                f"[yellow]Marked {result['entities_marked_stale']} entities as stale.[/yellow]"
            )
