"""hafiz chunks export — dump indexed chunks as JSON for agent-driven extraction."""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from sqlalchemy import func, select

from hafiz.core.database import Chunk, Entity, close_engine, get_session_factory

console = Console()


async def _export_chunks(
    *,
    project: str | None = None,
    path_prefix: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict]:
    """Export chunks as JSON-serializable dicts."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(Chunk).order_by(Chunk.indexed_at.desc())

        if project:
            stmt = stmt.where(Chunk.project == project)
        if path_prefix:
            stmt = stmt.where(Chunk.source_file.like(f"{path_prefix}%"))

        stmt = stmt.offset(offset).limit(limit)
        result = await session.execute(stmt)
        chunks = result.scalars().all()

        return [
            {
                "chunk_id": str(c.id),
                "content": c.content,
                "source_file": c.source_file,
                "language": c.language or "",
                "chunk_type": c.chunk_type,
                "line_start": c.line_start,
                "line_end": c.line_end,
            }
            for c in chunks
        ]


async def _count_chunks(
    *,
    project: str | None = None,
    path_prefix: str | None = None,
) -> int:
    """Count total chunks matching the filter."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(func.count(Chunk.id))
        if project:
            stmt = stmt.where(Chunk.project == project)
        if path_prefix:
            stmt = stmt.where(Chunk.source_file.like(f"{path_prefix}%"))
        result = await session.execute(stmt)
        return result.scalar() or 0


def run_chunks_export(
    *,
    project: str | None = None,
    path_prefix: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> None:
    """Export chunks as JSON to stdout."""

    async def _run():
        try:
            total = await _count_chunks(project=project, path_prefix=path_prefix)
            chunks = await _export_chunks(
                project=project,
                path_prefix=path_prefix,
                limit=limit,
                offset=offset,
            )
            output = {
                "total": total,
                "offset": offset,
                "limit": limit,
                "count": len(chunks),
                "chunks": chunks,
            }
            print(json.dumps(output, indent=2))
        finally:
            await close_engine()

    asyncio.run(_run())
