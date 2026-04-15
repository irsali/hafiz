"""hafiz chunks export — dump indexed chunks as JSON for agent-driven extraction."""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from sqlalchemy import func, select

from hafiz.core.database import Chunk, Entity, close_engine, get_session_factory

console = Console()


async def _files_with_entities(project: str | None = None) -> set[str]:
    """Return source files that already have entities extracted."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Entity.source_file).where(Entity.source_file.isnot(None))
        if project:
            stmt = stmt.where(Entity.project == project)
        result = await session.execute(stmt)
        return {row[0] for row in result.all()}


async def _export_chunks_by_file(
    *,
    project: str | None = None,
    path_prefix: str | None = None,
    unextracted: bool = False,
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Export chunks grouped by source file.

    Returns (files_list, total_chunk_count) where files_list is:
    [
        {
            "source_file": "src/auth.py",
            "language": "python",
            "chunks": [
                {"chunk_id": "...", "content": "...", "line_start": 1, "line_end": 50},
                ...
            ]
        },
        ...
    ]
    """
    session_factory = get_session_factory()

    extracted_files: set[str] = set()
    if unextracted:
        extracted_files = await _files_with_entities(project=project)

    async with session_factory() as session:
        # Count total matching chunks
        count_stmt = select(func.count(Chunk.id))
        if project:
            count_stmt = count_stmt.where(Chunk.project == project)
        if path_prefix:
            count_stmt = count_stmt.where(Chunk.source_file.like(f"{path_prefix}%"))
        if unextracted and extracted_files:
            count_stmt = count_stmt.where(Chunk.source_file.notin_(extracted_files))
        total = (await session.execute(count_stmt)).scalar() or 0

        # Fetch chunks ordered by file then line number
        stmt = select(Chunk).order_by(Chunk.source_file, Chunk.line_start)

        if project:
            stmt = stmt.where(Chunk.project == project)
        if path_prefix:
            stmt = stmt.where(Chunk.source_file.like(f"{path_prefix}%"))
        if unextracted and extracted_files:
            stmt = stmt.where(Chunk.source_file.notin_(extracted_files))

        stmt = stmt.offset(offset).limit(limit)
        result = await session.execute(stmt)
        chunks = result.scalars().all()

        # Group by source_file
        files_map: dict[str, dict] = {}
        for c in chunks:
            key = c.source_file or "(unknown)"
            if key not in files_map:
                files_map[key] = {
                    "source_file": c.source_file,
                    "language": c.language or "",
                    "chunks": [],
                }
            files_map[key]["chunks"].append({
                "chunk_id": str(c.id),
                "content": c.content,
                "line_start": c.line_start,
                "line_end": c.line_end,
            })

        return list(files_map.values()), total


def run_chunks_export(
    *,
    project: str | None = None,
    path_prefix: str | None = None,
    unextracted: bool = False,
    limit: int = 200,
    offset: int = 0,
) -> None:
    """Export chunks grouped by file as JSON to stdout."""

    async def _run():
        try:
            files, total = await _export_chunks_by_file(
                project=project,
                path_prefix=path_prefix,
                unextracted=unextracted,
                limit=limit,
                offset=offset,
            )
            chunk_count = sum(len(f["chunks"]) for f in files)
            output = {
                "total": total,
                "offset": offset,
                "limit": limit,
                "count": chunk_count,
                "files_count": len(files),
                "files": files,
            }
            if unextracted:
                output["filter"] = "unextracted"
            print(json.dumps(output, indent=2))
        finally:
            await close_engine()

    asyncio.run(_run())
