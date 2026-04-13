"""Store chunks into PostgreSQL via SQLAlchemy (direct control, no LlamaIndex vector store).

Handles upsert logic based on source_file + checksum for change detection.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select

from hafiz.core.database import Chunk, get_session_factory
from hafiz.core.chunker import ChunkResult


async def store_chunks(
    chunks: list[ChunkResult],
    embeddings: list[list[float]],
    *,
    project: str | None = None,
) -> int:
    """Store chunked content with embeddings into the database.

    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        async with session.begin():
            for chunk, embedding in zip(chunks, embeddings):
                db_chunk = Chunk(
                    id=uuid.uuid4(),
                    content=chunk.content,
                    embedding=embedding,
                    source_file=chunk.source_file,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    chunk_type=chunk.chunk_type,
                    language=chunk.language,
                    project=project or chunk.metadata.get("project"),
                    checksum=chunk.checksum,
                    indexed_at=datetime.now(timezone.utc),
                    metadata_=chunk.metadata,
                )
                session.add(db_chunk)
                stored += 1

    return stored


async def delete_chunks_for_file(source_file: str) -> int:
    """Delete all chunks for a given source file (for re-indexing).

    Returns the number of chunks deleted.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(
                delete(Chunk).where(Chunk.source_file == source_file)
            )
            return result.rowcount


async def file_checksums(source_file: str) -> set[str]:
    """Get the set of checksums for chunks of a given file."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(Chunk.checksum).where(Chunk.source_file == source_file)
        )
        return {row[0] for row in result.fetchall() if row[0]}


async def get_indexed_files() -> dict[str, set[str]]:
    """Get a mapping of source_file -> set of checksums for all indexed files."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(Chunk.source_file, Chunk.checksum)
        )
        files: dict[str, set[str]] = {}
        for source_file, checksum in result.fetchall():
            files.setdefault(source_file, set())
            if checksum:
                files[source_file].add(checksum)
        return files
