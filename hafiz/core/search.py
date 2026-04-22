"""Vector similarity search over the structural-grounding schema.

Queries hit the ``embeddings`` table and join back to
``unit_revisions → units → files`` for context. Only current revisions
(``superseded_at IS NULL``) are searched — old bodies stay in the DB for
history but don't pollute retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select

from hafiz.core.database import (
    Embedding,
    File,
    Unit,
    UnitRevision,
    get_session_factory,
)
from hafiz.core.embeddings import embed_query


@dataclass
class SearchResult:
    """A single search result. Fields are populated from the joined
    ``embeddings`` → ``unit_revisions`` → ``units`` → ``files`` row."""

    id: str                       # embedding row id
    unit_id: str
    unit_name: str
    kind: str                     # namespaced: code.function, doc.heading, file.raw, …
    content: str                  # the embedded part (may be a slice of the unit body)
    source_file: str
    line_start: int | None
    line_end: int | None
    language: str | None
    project: str | None
    part_index: int
    score: float
    is_neighbor: bool = False


async def vector_search(
    query: str,
    *,
    limit: int = 10,
    project: str | list[str] | None = None,
    kind: str | None = None,
    similarity_threshold: float = 0.0,
) -> list[SearchResult]:
    """Search embeddings by cosine similarity and return enriched results.

    Args:
        query: Search query text (will be embedded).
        limit: Maximum results.
        project: Filter by project name (str), multiple projects (list),
            or None for no project filter.
        kind: Filter by unit kind (e.g. ``"code.function"``,
            ``"doc.heading"``). Matches on ``units.kind``.
        similarity_threshold: Drop results below this cosine similarity.

    Returns:
        List of SearchResult ordered by similarity (best first).
    """
    query_embedding = await embed_query(query)

    similarity = (
        1 - Embedding.embedding.cosine_distance(query_embedding)
    ).label("similarity")

    stmt = (
        select(Embedding, UnitRevision, Unit, File, similarity)
        .join(UnitRevision, UnitRevision.id == Embedding.unit_revision_id)
        .join(Unit, Unit.id == UnitRevision.unit_id)
        .join(File, File.id == Unit.file_id)
        .where(Embedding.embedding.isnot(None))
        .where(UnitRevision.superseded_at.is_(None))
        .where(Unit.valid_until.is_(None))
        .where(File.valid_until.is_(None))
        .order_by(Embedding.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )

    if isinstance(project, list):
        stmt = stmt.where(File.project.in_(project))
    elif project:
        stmt = stmt.where(File.project == project)
    if kind:
        stmt = stmt.where(Unit.kind == kind)

    session_factory = get_session_factory()
    async with session_factory() as session:
        rows = (await session.execute(stmt)).all()

    results: list[SearchResult] = []
    for emb, rev, unit, file, sim in rows:
        sim_f = float(sim)
        if sim_f < similarity_threshold:
            continue
        results.append(
            SearchResult(
                id=str(emb.id),
                unit_id=str(unit.id),
                unit_name=unit.name,
                kind=unit.kind,
                content=emb.content,
                source_file=file.path,
                line_start=rev.line_start,
                line_end=rev.line_end,
                language=file.language,
                project=file.project,
                part_index=emb.part_index,
                score=round(sim_f, 4),
            )
        )
    return results


async def count_embeddings(project: str | None = None) -> int:
    """Count current embedding rows, optionally filtered by project."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(func.count())
            .select_from(Embedding)
            .join(UnitRevision, UnitRevision.id == Embedding.unit_revision_id)
            .join(Unit, Unit.id == UnitRevision.unit_id)
            .join(File, File.id == Unit.file_id)
            .where(UnitRevision.superseded_at.is_(None))
            .where(Unit.valid_until.is_(None))
            .where(File.valid_until.is_(None))
        )
        if project:
            stmt = stmt.where(File.project == project)
        return (await session.execute(stmt)).scalar() or 0
