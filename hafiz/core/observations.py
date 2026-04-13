"""Observation storage and retrieval with vector similarity search.

Store high-level decisions, facts, and learnings as embeddings for semantic recall.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, func, update

from hafiz.core.database import Observation, get_session_factory
from hafiz.core.embeddings import embed_query


@dataclass
class ObservationResult:
    """A single observation search result with similarity score."""

    id: str
    content: str
    obs_type: str
    source: str | None
    project: str | None
    tags: list[str] | None
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    metadata: dict
    score: float


async def store_observation(
    content: str,
    *,
    obs_type: str = "fact",
    source: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    metadata: dict | None = None,
) -> Observation:
    """Store a new observation with its embedding.

    Args:
        content: The observation text.
        obs_type: Type — fact, decision, learning, pattern, warning.
        source: Origin (e.g. "agent:bilal", "user:manual").
        project: Project name.
        tags: Categorization tags.
        confidence: Confidence score 0.0–1.0.
        valid_from: When the observation becomes valid (default: now).
        valid_until: When the observation expires (None = forever).
        metadata: Arbitrary JSONB metadata.

    Returns:
        The stored Observation ORM object.
    """
    embedding = await embed_query(content)

    now = datetime.now(timezone.utc)
    obs = Observation(
        id=uuid.uuid4(),
        content=content,
        embedding=embedding,
        obs_type=obs_type,
        source=source,
        project=project,
        tags=tags,
        confidence=confidence,
        valid_from=valid_from or now,
        valid_until=valid_until,
        metadata_=metadata or {},
    )

    session_factory = get_session_factory()
    async with session_factory() as session:
        session.add(obs)
        await session.commit()
        await session.refresh(obs)
        return obs


async def search_observations(
    query: str,
    *,
    limit: int = 10,
    project: str | None = None,
    obs_type: str | None = None,
    active_only: bool = True,
) -> list[ObservationResult]:
    """Search observations by vector similarity.

    Args:
        query: The search query text.
        limit: Maximum number of results.
        project: Filter by project name.
        obs_type: Filter by observation type.
        active_only: Only return currently valid observations.

    Returns:
        List of ObservationResult sorted by similarity (highest first).
    """
    query_embedding = await embed_query(query)

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(
                Observation,
                (1 - Observation.embedding.cosine_distance(query_embedding)).label(
                    "similarity"
                ),
            )
            .where(Observation.embedding.isnot(None))
            .order_by(Observation.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        if project:
            stmt = stmt.where(Observation.project == project)
        if obs_type:
            stmt = stmt.where(Observation.obs_type == obs_type)
        if active_only:
            now = datetime.now(timezone.utc)
            stmt = stmt.where(Observation.valid_from <= now)
            stmt = stmt.where(
                (Observation.valid_until.is_(None)) | (Observation.valid_until > now)
            )

        result = await session.execute(stmt)
        rows = result.all()

        return [
            ObservationResult(
                id=str(obs.id),
                content=obs.content,
                obs_type=obs.obs_type,
                source=obs.source,
                project=obs.project,
                tags=obs.tags,
                confidence=obs.confidence,
                valid_from=obs.valid_from,
                valid_until=obs.valid_until,
                metadata=obs.metadata_ or {},
                score=round(float(similarity), 4),
            )
            for obs, similarity in rows
        ]


async def list_observations(
    *,
    project: str | None = None,
    obs_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Observation]:
    """List observations with optional filters.

    Args:
        project: Filter by project name.
        obs_type: Filter by observation type.
        limit: Maximum number of results.
        offset: Skip this many results.

    Returns:
        List of Observation ORM objects.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(Observation)
            .order_by(Observation.valid_from.desc())
            .limit(limit)
            .offset(offset)
        )
        if project:
            stmt = stmt.where(Observation.project == project)
        if obs_type:
            stmt = stmt.where(Observation.obs_type == obs_type)

        result = await session.execute(stmt)
        return list(result.scalars().all())


async def update_observation(
    obs_id: str,
    *,
    content: str | None = None,
    obs_type: str | None = None,
    confidence: float | None = None,
    valid_until: datetime | None = None,
) -> Observation | None:
    """Update an observation. Re-embeds if content changes.

    Args:
        obs_id: UUID of the observation.
        content: New content (triggers re-embedding).
        obs_type: New observation type.
        confidence: New confidence score.
        valid_until: New expiration datetime.

    Returns:
        The updated Observation, or None if not found.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Observation).where(Observation.id == uuid.UUID(obs_id))
        result = await session.execute(stmt)
        obs = result.scalar_one_or_none()

        if obs is None:
            return None

        if content is not None and content != obs.content:
            obs.content = content
            obs.embedding = await embed_query(content)
        if obs_type is not None:
            obs.obs_type = obs_type
        if confidence is not None:
            obs.confidence = confidence
        if valid_until is not None:
            obs.valid_until = valid_until

        await session.commit()
        await session.refresh(obs)
        return obs


async def invalidate_observation(obs_id: str) -> Observation | None:
    """Invalidate an observation by setting valid_until = now.

    Args:
        obs_id: UUID of the observation.

    Returns:
        The updated Observation, or None if not found.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Observation).where(Observation.id == uuid.UUID(obs_id))
        result = await session.execute(stmt)
        obs = result.scalar_one_or_none()

        if obs is None:
            return None

        obs.valid_until = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(obs)
        return obs
