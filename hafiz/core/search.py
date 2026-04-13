"""Vector similarity search using pgvector cosine distance.

Direct SQL queries against the chunks table — no LlamaIndex vector store abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, text

from hafiz.core.database import Chunk, get_session_factory
from hafiz.core.embeddings import embed_query


@dataclass
class SearchResult:
    """A single search result with similarity score."""

    id: str
    content: str
    source_file: str
    line_start: int | None
    line_end: int | None
    chunk_type: str
    language: str | None
    project: str | None
    score: float
    metadata: dict


async def vector_search(
    query: str,
    *,
    limit: int = 10,
    project: str | None = None,
    chunk_type: str | None = None,
    similarity_threshold: float = 0.0,
) -> list[SearchResult]:
    """Search chunks by vector similarity using cosine distance.

    Args:
        query: The search query text.
        limit: Maximum number of results.
        project: Filter by project name.
        chunk_type: Filter by chunk type (code, doc, note, decision).
        similarity_threshold: Minimum similarity score (0-1).

    Returns:
        List of SearchResult sorted by similarity (highest first).
    """
    query_embedding = await embed_query(query)

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Build the query with cosine distance
        # pgvector: 1 - (embedding <=> query_embedding) gives cosine similarity
        stmt = (
            select(
                Chunk,
                (1 - Chunk.embedding.cosine_distance(query_embedding)).label("similarity"),
            )
            .where(Chunk.embedding.isnot(None))
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        # Apply filters
        if project:
            stmt = stmt.where(Chunk.project == project)
        if chunk_type:
            stmt = stmt.where(Chunk.chunk_type == chunk_type)

        result = await session.execute(stmt)
        rows = result.all()

        results = []
        for chunk, similarity in rows:
            if similarity < similarity_threshold:
                continue
            results.append(
                SearchResult(
                    id=str(chunk.id),
                    content=chunk.content,
                    source_file=chunk.source_file,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    chunk_type=chunk.chunk_type,
                    language=chunk.language,
                    project=chunk.project,
                    score=round(float(similarity), 4),
                    metadata=chunk.metadata_ or {},
                )
            )

        return results


async def count_chunks(project: str | None = None) -> int:
    """Count total chunks, optionally filtered by project."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        from sqlalchemy import func

        stmt = select(func.count()).select_from(Chunk)
        if project:
            stmt = stmt.where(Chunk.project == project)
        result = await session.execute(stmt)
        return result.scalar() or 0
